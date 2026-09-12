import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SERIALIZATION_CHOICES = ("text_template", "json", "llm")
DEFAULT_DATASETS = ("hepatitis", "heart", "diabetes")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent


def config_path_from_dataset(dataset_name):
    """
    Build the expected config path for a dataset name.
    """
    return CONFIG_DIR / f"{dataset_name}_config.json"


def resolve_repo_path(path):
    """Resolve config paths consistently, regardless of the current directory."""
    path = Path(path)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_config(config_path):
    """
    Load the dataset processing configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Configuration dictionary containing dataset path, target column,
        output path, column mappings, numerical columns, and optional loading
        settings such as format, block size, used indices, and column names.
    """
    with Path(config_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_category_value(x):
    """
    Clean a categorical value and convert it to a string key.

    This function handles missing values, question marks, and values such as
    1, 1.0, or 1.2 by converting them to integer category keys. The result is
    returned as a string because JSON mapping keys are strings.

    Parameters
    ----------
    x : any
        Raw categorical value from the dataset.

    Returns
    -------
    str or numpy.nan
        Cleaned category value as a string, or np.nan if the value is missing.
    """
    if pd.isna(x) or x == "?":
        return np.nan

    return str(int(float(x)))


def clean_numeric_value(x):
    """
    Clean a numeric value and convert it to float.

    Parameters
    ----------
    x : any
        Raw numeric value from the dataset.

    Returns
    -------
    float or numpy.nan
        Numeric value as a float, or np.nan if the value is missing.
    """
    if pd.isna(x) or x == "?":
        return np.nan

    return float(x)


def map_continuous_category(x, mapping):
    """
    Map a continuous value to the nearest categorical prototype.

    Example:
        1.7 -> 2 -> "more"
        0.8 -> 1 -> "typical"
    """
    if pd.isna(x) or x == "?":
        return "unknown"

    x = float(x)

    nearest = min(
        mapping.keys(),
        key=lambda k: abs(x - float(k))
    )

    return mapping[nearest]


def load_dataframe(config, filepath=None):
    """
    Load a dataset according to the format specified in the config file.

    The preprocessing notebooks export ordinary CSV files with headers. A
    headerless CSV can also be loaded when ``has_header`` is false and
    ``column_names`` is configured.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Common keys are:
        - filepath : str
            Path to the dataset file.
        - has_header : bool
            Whether the CSV file already contains column names.
            Defaults to True.
        - column_names : list of str
            Column names to use when has_header is False.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe ready for cleaning and serialization.
    """

    filepath = resolve_repo_path(filepath or config["filepath"])

    if config.get("has_header", True):
        return pd.read_csv(filepath)

    return pd.read_csv(
        filepath,
        header=None,
        names=config["column_names"]
    )


def serialize_row(row, target_col):
    """
    Convert one dataframe row into a natural-language text serialization.

    Each non-target column is converted into a sentence of the form:
    "The column_name is value."

    Missing values are serialized as "unknown".

    Parameters
    ----------
    row : pandas.Series
        One row from the dataframe.

    target_col : str
        Name of the target column to exclude from serialization.

    Returns
    -------
    str
        Serialized text representation of the row.
    """
    sentences = []

    for col, value in row.items():
        if col == target_col:
            continue

        if pd.isna(value):
            value = "unknown"

        sentences.append(f"The {col} is {value}.")

    return " ".join(sentences)


def serialization_value(value):
    """
    Convert a dataframe value into a stable serialization value.
    """
    if pd.isna(value):
        return "unknown"

    if isinstance(value, np.generic):
        return value.item()

    return value


def serialize_row_json(row, target_col):
    """
    Convert one dataframe row into one-line JSON serialization.

    Each non-target column is serialized as a JSON key-value pair. Missing
    values are written as "unknown" to match the text-template behavior.
    """
    row_dict = {}

    for col, value in row.items():
        if col == target_col or col.startswith("serialized_"):
            continue

        row_dict[col] = serialization_value(value)

    return json.dumps(row_dict, ensure_ascii=True)


def serialize_row_llm_style(row, target_col):
    """
    Convert one dataframe row into an LLM-style natural-language sentence.

    This deterministic fallback keeps cluster preprocessing self-contained.
    If you generate natural-language rows with a separate LLM pass, store them
    in the same `serialized_llm` column and the experiment runner will use
    them unchanged.
    """
    phrases = []

    for col, value in row.items():
        if col == target_col or col.startswith("serialized_"):
            continue

        phrases.append(f"{col} {serialization_value(value)}")

    return f"The patient record describes a person with {', '.join(phrases)}."


def build_llm_serialization_prompt(row, target_col):
    """
    Build the prompt used to ask an LLM for natural-language row serialization.
    """
    feature_dict = {}

    for col, value in row.items():
        if col == target_col or col.startswith("serialized_"):
            continue

        feature_dict[col] = serialization_value(value)

    feature_json = json.dumps(feature_dict, ensure_ascii=True)

    return f"""
Convert this tabular medical patient record into one concise natural-language sentence.
Do not infer or mention the target label. Preserve all feature values.

Record:
{feature_json}

Sentence:
""".strip()


def generate_llm_serialization(row, target_col, tokenizer, model, max_new_tokens):
    """
    Generate one natural-language row serialization with an LLM.
    """
    import torch

    prompt = build_llm_serialization_prompt(row, target_col)
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def load_llm_serializer(model_name):
    """
    Load the optional model used to create LLM serializations.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    return tokenizer, model


def process_dataset(
    config,
    filepath=None,
    output_file=None,
    serializations=SERIALIZATION_CHOICES,
    llm_tokenizer=None,
    llm_model=None,
    llm_max_new_tokens=80,
):
    """
    Load, clean, serialize, and save a tabular dataset.

    This function is dataset-general. The behavior is controlled by the config
    file, so the same code can process both the Hepatitis dataset and the Heart
    dataset.

    Processing steps
    ----------------
    1. Load the dataset using `load_dataframe`.
    2. Clean and map the target column.
    3. Clean and map categorical columns.
    4. Clean numerical columns.
    5. Serialize each row into text.
    6. Save the serialized text and target label to a CSV file.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys such as:
        - input_files : dict
            Train/test input paths used by :func:`serialize`.
        - output_files : dict
            Matching train/test serialized output paths.
        - target_col : str
            Name of the target column.
        - target_mapping : dict
            Mapping from raw target values to target labels.
        - column_mappings : dict
            Mapping rules for categorical columns.
        - numeric_cols : list of str
            Names of numerical columns to clean.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe containing the cleaned columns and the
        `serialized_text` column.
    """
    df = load_dataframe(config, filepath=filepath)

    df = df.drop(columns=config.get("drop_columns", []), errors="ignore")

    target_col = config["target_col"]

    raw_target = df[target_col].copy()
    df[target_col] = (
        raw_target
        .apply(clean_category_value)
        .map(config["target_mapping"])
    )

    if df[target_col].isna().any():
        bad_values = sorted(
            raw_target.loc[df[target_col].isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Unmapped values in target column '{target_col}': {bad_values}")

    # Give every serialized dataset the same target-column contract.
    df = df.rename(columns={target_col: "target_col"})
    target_col = "target_col"

    for col, mapping in config.get("column_mappings", {}).items():
        df[col] = (
            df[col]
            .apply(clean_category_value)
            .map(mapping)
            .fillna("unknown")
        )

    for col, mapping in config.get("continuous_categorical_mappings", {}).items():
        df[col] = df[col].apply(lambda x: map_continuous_category(x, mapping))

    for col in config.get("numeric_cols", []):
        df[col] = df[col].apply(clean_numeric_value)

    output_columns = []

    if "text_template" in serializations:
        df["serialized_text"] = df.apply(
            lambda row: serialize_row(row, target_col),
            axis=1
        )
        output_columns.append("serialized_text")

    if "json" in serializations:
        df["serialized_json"] = df.apply(
            lambda row: serialize_row_json(row, target_col),
            axis=1
        )
        output_columns.append("serialized_json")

    if "llm" in serializations:
        if llm_tokenizer is not None and llm_model is not None:
            df["serialized_llm"] = df.apply(
                lambda row: generate_llm_serialization(
                    row,
                    target_col,
                    llm_tokenizer,
                    llm_model,
                    llm_max_new_tokens,
                ),
                axis=1
            )
        else:
            df["serialized_llm"] = df.apply(
                lambda row: serialize_row_llm_style(row, target_col),
                axis=1
            )
        output_columns.append("serialized_llm")

    output_columns.append(target_col)

    output_file = resolve_repo_path(output_file or config["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(output_file, index=False)

    return df


def serialize(
    config_path,
    serializations=SERIALIZATION_CHOICES,
    llm_tokenizer=None,
    llm_model=None,
    llm_max_new_tokens=80,
):
    """
    Main function to process the dataset based on the provided configuration.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    pandas.DataFrame
        Processed dataframe containing the cleaned columns and the
        `serialized_text` column.
    """
    config = load_config(config_path)

    input_files = config.get("input_files")
    output_files = config.get("output_files")

    if input_files is None or output_files is None:
        return {
            "dataset": process_dataset(
                config,
                serializations=serializations,
                llm_tokenizer=llm_tokenizer,
                llm_model=llm_model,
                llm_max_new_tokens=llm_max_new_tokens,
            )
        }

    if set(input_files) != set(output_files):
        raise ValueError("input_files and output_files must define the same splits.")

    return {
        split: process_dataset(
            config,
            filepath=input_files[split],
            output_file=output_files[split],
            serializations=serializations,
            llm_tokenizer=llm_tokenizer,
            llm_model=llm_model,
            llm_max_new_tokens=llm_max_new_tokens,
        )
        for split in input_files
    }


def run_serialization(
    datasets=None,
    configs=None,
    serializations=None,
    llm_model=None,
    llm_max_new_tokens=80,
):
    """
    Serialize configured datasets from Python code or notebooks.
    """
    if configs is None:
        if datasets is None:
            datasets = list(DEFAULT_DATASETS)

        configs = [
            config_path_from_dataset(dataset_name)
            for dataset_name in datasets
        ]

    if serializations is None:
        serializations = list(SERIALIZATION_CHOICES)

    llm_tokenizer = None
    loaded_llm_model = None

    if llm_model is not None and "llm" in serializations:
        print(f"Loading LLM serializer: {llm_model}")
        llm_tokenizer, loaded_llm_model = load_llm_serializer(llm_model)

    serialized_datasets = {}

    for config_path in configs:
        split_frames = serialize(
            config_path,
            serializations=serializations,
            llm_tokenizer=llm_tokenizer,
            llm_model=loaded_llm_model,
            llm_max_new_tokens=llm_max_new_tokens,
        )
        serialized_datasets[str(config_path)] = split_frames
        for split, df in split_frames.items():
            print(
                f"Serialized {config_path} [{split}]: {len(df)} rows "
                f"({', '.join(serializations)})"
            )

    return serialized_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to serialize. Uses models/tabllm/{dataset_name}_config.json.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional explicit config file paths. If provided, overrides --datasets.",
    )
    parser.add_argument(
        "--serializations",
        nargs="+",
        default=list(SERIALIZATION_CHOICES),
        choices=SERIALIZATION_CHOICES,
        help="Serialization styles to generate.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional Hugging Face model used to generate serialized_llm.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=80,
        help="Maximum new tokens for each generated LLM serialization.",
    )

    args = parser.parse_args()

    run_serialization(
        datasets=args.datasets,
        configs=args.configs,
        serializations=args.serializations,
        llm_model=args.llm_model,
        llm_max_new_tokens=args.llm_max_new_tokens,
    )


if __name__ == "__main__":
    main()
