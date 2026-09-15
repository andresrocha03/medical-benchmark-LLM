import os
from collections.abc import Sequence
from typing import Any

from config.setup_config import PREDICTION_LABELS, RESULTS_DIR


SERIALIZATION_COLUMNS = {
    "text_template": "serialized_text",
    "json": "serialized_json",
    "llm": "serialized_llm",
}


def load_model(model_name: str) -> tuple[Any, Any]:
    """Load a Hugging Face tokenizer and causal language model.

    input:
        - model_name: str

    output:
        - tokenizer_and_model: tuple[transformers.PreTrainedTokenizer, transformers.PreTrainedModel]
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


def build_prompt(
    serialized_row: str,
    task: str,
    target_labels: Sequence[str],
    positive_target_label: str,
) -> str:
    """Build the classification prompt for a serialized patient row.

    input:
        - serialized_row: str
        - task: str
        - target_labels: list[str]
        - positive_target_label: str

    output:
        - prompt: str
    """
    negative_target_labels = [
        label for label in target_labels if label != positive_target_label
    ]
    if len(target_labels) != 2 or len(negative_target_labels) != 1:
        raise ValueError(
            "TabLLM requires exactly two target labels and one positive label."
        )

    negative_target_label = negative_target_labels[0]
    negative_response, positive_response = PREDICTION_LABELS

    return f"""
You are performing a binary medical classification task.

Input:
{serialized_row}

Task:
{task}

Label mapping:
- Answer "{positive_response}" when the predicted outcome is "{positive_target_label}".
- Answer "{negative_response}" when the predicted outcome is "{negative_target_label}".

Rules:
- Choose exactly one of these labels: "{positive_response}" or "{negative_response}".
- Do not explain your reasoning.
- Do not output any text other than the label.
- If uncertain, choose the most likely label.

Answer:
""".strip()


def predict_one_row(
    serialized_row: str,
    task: str,
    target_labels: Sequence[str],
    positive_target_label: str,
    tokenizer: Any,
    model: Any,
) -> dict[str, str]:
    """Generate one model response for a serialized row.

    input:
        - serialized_row: str
        - task: str
        - target_labels: list[str]
        - positive_target_label: str
        - tokenizer: transformers.PreTrainedTokenizer
        - model: transformers.PreTrainedModel

    output:
        - prediction_result: dict[str, str]
    """
    import torch

    prompt = build_prompt(
        serialized_row,
        task,
        target_labels,
        positive_target_label,
    )
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


def predict_batch(
    serialized_rows: Sequence[str],
    task: str,
    target_labels: Sequence[str],
    positive_target_label: str,
    tokenizer: Any,
    model: Any,
) -> list[dict[str, str]]:
    """Generate model responses for a batch of serialized rows.

    input:
        - serialized_rows: list[str]
        - task: str
        - target_labels: list[str]
        - positive_target_label: str
        - tokenizer: transformers.PreTrainedTokenizer
        - model: transformers.PreTrainedModel

    output:
        - prediction_results: list[dict[str, str]]
    """
    import torch

    prompts = [
        build_prompt(
            serialized_row,
            task,
            target_labels,
            positive_target_label,
        )
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
    dataset_name: str,
    dataset_config: dict[str, Any],
    model_key: str,
    tokenizer: Any,
    model: Any,
    serialization_style: str = "text_template",
    max_examples: int | None = None,
    run_tag: str | None = None,
    batch_size: int = 1,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run one model on one dataset and save its raw responses.

    input:
        - dataset_name: str
        - dataset_config: dict
        - model_key: str
        - tokenizer: transformers.PreTrainedTokenizer
        - model: transformers.PreTrainedModel
        - serialization_style: str
        - max_examples: int | None
        - run_tag: str | None
        - batch_size: int
        - model_name: str | None

    output:
        - run_metadata: dict
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

    displayed_model = (
        f"{model_name} ({model_key})"
        if model_name and model_name != model_key
        else model_key
    )
    print(
        f"\nModel: {displayed_model} | Dataset: {dataset_name} "
        f"| Serialization: {serialization_style} | Rows: {len(df)} "
        f"| Batch size: {batch_size}",
        flush=True,
    )

    serialized_rows = df[serialization_column].tolist()
    next_progress_row = 150

    for start_idx in range(0, len(serialized_rows), batch_size):
        batch_rows = serialized_rows[start_idx:start_idx + batch_size]

        if batch_size == 1:
            batch_results = [
                predict_one_row(
                    serialized_row=batch_rows[0],
                    task=dataset_config["task"],
                    target_labels=dataset_config["choices"],
                    positive_target_label=dataset_config["positive_label"],
                    tokenizer=tokenizer,
                    model=model,
                )
            ]
        else:
            batch_results = predict_batch(
                serialized_rows=batch_rows,
                task=dataset_config["task"],
                target_labels=dataset_config["choices"],
                positive_target_label=dataset_config["positive_label"],
                tokenizer=tokenizer,
                model=model,
            )

        for result in batch_results:
            raw_responses.append(result["output"])
            prompts.append(result["prompt"])

        processed_rows = min(start_idx + len(batch_rows), len(serialized_rows))
        while processed_rows >= next_progress_row:
            print(
                f"  Processed {next_progress_row}/{len(serialized_rows)} rows",
                flush=True,
            )
            next_progress_row += 150

        if (
            processed_rows == len(serialized_rows)
            and processed_rows % 150 != 0
        ):
            print(
                f"  Processed {processed_rows}/{len(serialized_rows)} rows",
                flush=True,
            )

    df["output"] = raw_responses
    df["prompt"] = prompts

    output_path = os.path.join(
        dataset_dir,
        f"{model_key}_{run_tag}_raw_responses.csv"
        if run_tag
        else f"{model_key}_raw_responses.csv",
    )

    df.to_csv(output_path, index=False)

    run_metadata = {
        "dataset": dataset_name,
        "serialization": serialization_style,
        "model": model_key,
        "n_examples": int(len(df)),
        "total_examples": int(total_examples),
        "run_tag": run_tag,
        "batch_size": int(batch_size),
        "output_file": output_path,
    }

    return run_metadata


def clear_model_resources(model: Any, tokenizer: Any) -> None:
    """Release model references and clear CUDA memory when available.

    input:
        - model: transformers.PreTrainedModel
        - tokenizer: transformers.PreTrainedTokenizer

    output:
        - None: None
    """
    import gc
    import torch

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
