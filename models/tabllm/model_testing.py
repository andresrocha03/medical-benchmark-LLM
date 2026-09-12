import json
import os
import re

try:
    from .setup_config import RESULTS_DIR
except ImportError:  # Allow direct execution from models/tabllm.
    from setup_config import RESULTS_DIR


SERIALIZATION_COLUMNS = {
    "text_template": "serialized_text",
    "json": "serialized_json",
    "llm": "serialized_llm",
}


def safe_name(name):
    """
    Convert a string into a safe filename.

    Parameters
    ----------
    name : str
        Original string, such as a model name or dataset name.

    Returns
    -------
    str
        Filename-safe version of the input string.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def load_model(model_name):
    """
    Load a Hugging Face tokenizer and causal language model.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.

    Returns
    -------
    tuple
        A tuple containing:
        - tokenizer : transformers.PreTrainedTokenizer
            Loaded tokenizer.
        - model : transformers.PreTrainedModel
            Loaded causal language model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        device_names = [
            torch.cuda.get_device_name(device_idx)
            for device_idx in range(torch.cuda.device_count())
        ]
        print(f"Using GPU: {', '.join(device_names)}")
    else:
        print("Using CPU")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.eval()

    return tokenizer, model


def build_prompt(serialized_row, task, choices):
    """
    Build the classification prompt for a serialized patient row.

    Parameters
    ----------
    serialized_row : str
        Serialized patient information.

    task : str
        Dataset-specific classification task.

    choices : list of str
        Allowed class labels.

    Returns
    -------
    str
        Prompt to send to the language model.
    """
    choices_text = "\n".join([f"- {choice}" for choice in choices])

    return f"""
You are performing a supervised machine learning classification task.

Input:
{serialized_row}

Task:
{task}

Valid labels:
{choices_text}

Rules:
- Choose exactly one label.
- You must output one of the valid labels.
- Do not explain your reasoning.
- Do not output any text other than the label.
- If uncertain, choose the most likely label.

Answer:
""".strip()


def predict_one_row(serialized_row, task, choices, tokenizer, model):
    """
    Generate one model response for a row.

    Parameters
    ----------
    serialized_row : str
        Serialized patient data.

    task : str
        Dataset-specific task instruction.

    choices : list of str
        Allowed output labels.

    tokenizer : transformers.PreTrainedTokenizer
        Loaded tokenizer.

    model : transformers.PreTrainedModel
        Loaded model.

    Returns
    -------
    dict
        Dictionary containing:
        - output : str
            Full model-generated response.
        - prompt : str
            Full prompt sent to the model.
    """
    import torch

    prompt = build_prompt(serialized_row, task, choices)
    messages = [{"role": "user", "content": prompt}]

    if tokenizer.chat_template is None:
        input_text = prompt
    else:
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=28,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw_response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    return {
        "output": raw_response,
        "prompt": prompt,
    }


def predict_batch(serialized_rows, task, choices, tokenizer, model):
    """
    Generate model responses for a batch of serialized rows.
    """
    import torch

    prompts = [
        build_prompt(serialized_row, task, choices)
        for serialized_row in serialized_rows
    ]

    input_texts = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]

        if tokenizer.chat_template is None:
            input_text = prompt
        else:
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        input_texts.append(input_text)

    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=28,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    raw_responses = [
        tokenizer.decode(
            output[prompt_length:],
            skip_special_tokens=True,
        ).strip()
        for output in outputs
    ]

    return [
        {
            "output": raw_response,
            "prompt": prompt,
        }
        for raw_response, prompt in zip(raw_responses, prompts)
    ]


def run_dataset(
    dataset_name,
    dataset_config,
    model_key,
    tokenizer,
    model,
    serialization_style="text_template",
    max_examples=None,
    run_tag=None,
    batch_size=1,
):
    """
    Run one model on one dataset and save raw responses.

    Parameters
    ----------
    dataset_name : str
        Dataset name.

    dataset_config : dict
        Dataset configuration containing path, task, and choices.

    model_key : str
        Short model name used for saving files.

    tokenizer : transformers.PreTrainedTokenizer
        Loaded tokenizer.

    model : transformers.PreTrainedModel
        Loaded model.

    serialization_style : str
        Serialization style to use as model input. Supported values are:
        "text_template", "json", and "llm".

    max_examples : int, optional
        If provided, only the first `max_examples` rows are processed.

    run_tag : str, optional
        Optional tag added to output filenames, such as "test".

    batch_size : int
        Number of rows to generate at once. Defaults to 1.

    Returns
    -------
    dict
        Summary dictionary containing dataset name, model name,
        number of examples, and output file path.
    """
    import pandas as pd

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    # Zero-shot evaluation must never read the serialized training split.
    df = pd.read_csv(dataset_config["test_path"])
    total_examples = len(df)

    if max_examples is not None:
        df = df.head(max_examples).copy()

    serialization_column = SERIALIZATION_COLUMNS[serialization_style]

    if serialization_column not in df.columns:
        if serialization_style == "text_template" and "serialized_text" in df.columns:
            serialization_column = "serialized_text"
        else:
            raise ValueError(
                f"Missing column '{serialization_column}' in {dataset_config['test_path']}. "
                "Regenerate the serialized datasets with serialization.py."
            )

    dataset_dir = os.path.join(RESULTS_DIR, dataset_name, serialization_style)
    os.makedirs(dataset_dir, exist_ok=True)

    raw_responses = []
    prompts = []

    print(
        f"[{dataset_name} | {serialization_style} | {model_key}] "
        f"Running {len(df)} examples with batch_size={batch_size}..."
    )

    serialized_rows = df[serialization_column].tolist()

    for start_idx in range(0, len(serialized_rows), batch_size):
        batch_rows = serialized_rows[start_idx:start_idx + batch_size]

        if batch_size == 1:
            batch_results = [
                predict_one_row(
                    serialized_row=batch_rows[0],
                    task=dataset_config["task"],
                    choices=dataset_config["choices"],
                    tokenizer=tokenizer,
                    model=model,
                )
            ]
        else:
            batch_results = predict_batch(
                serialized_rows=batch_rows,
                task=dataset_config["task"],
                choices=dataset_config["choices"],
                tokenizer=tokenizer,
                model=model,
            )

        for result in batch_results:
            raw_responses.append(result["output"])
            prompts.append(result["prompt"])

    df["output"] = raw_responses
    df["prompt"] = prompts

    output_path = os.path.join(
        dataset_dir,
        f"{model_key}_{run_tag}_raw_responses.csv"
        if run_tag
        else f"{model_key}_raw_responses.csv",
    )

    df.to_csv(output_path, index=False)

    metrics = {
        "dataset": dataset_name,
        "serialization": serialization_style,
        "model": model_key,
        "n_examples": int(len(df)),
        "total_examples": int(total_examples),
        "run_tag": run_tag,
        "batch_size": int(batch_size),
        "output_file": output_path,
    }

    metrics_path = os.path.join(
        dataset_dir,
        f"{model_key}_{run_tag}_metrics.json"
        if run_tag
        else f"{model_key}_metrics.json",
    )

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    return metrics


def clear_model_resources(model, tokenizer):
    """
    Release model references and clear CUDA memory when available.
    """
    import gc
    import torch

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
