import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SERIALIZATION_CHOICES = ("text_template", "json", "llm")
DEFAULT_DATASETS = ("hepatitis", "heart", "diabetes")
PROGRESS_INTERVAL = 150
CONFIG_DIR = Path("../../../models/tabllm/config")


def config_path_from_dataset(dataset_name: str) -> Path:
    """Build the expected configuration path for a dataset name.

    input:
        - dataset_name: str

    output:
        - config_path: pathlib.Path
    """
    return CONFIG_DIR / f"{dataset_name}_config.json"


def resolve_configured_path(path: str | Path) -> Path:
    """Resolve a configured project path from the serialization directory.

    input:
        - path: str | pathlib.Path

    output:
        - configured_path: pathlib.Path
    """
    path = Path(path)
    return path if path.is_absolute() else Path("../../..") / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a dataset processing configuration from JSON.

    input:
        - config_path: str | pathlib.Path

    output:
        - config: dict[str, Any]
    """
    with Path(config_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_category_value(x: Any) -> str | float:
    """Convert a categorical value to a mapping key or NaN.

    input:
        - x: Any

    output:
        - cleaned_value: str | float
    """
    if pd.isna(x) or x == "?":
        return np.nan

    return str(int(float(x)))


def clean_numeric_value(x: Any) -> float:
    """Convert a numeric value to a float or NaN.

    input:
        - x: Any

    output:
        - cleaned_value: float
    """
    if pd.isna(x) or x == "?":
        return np.nan

    return float(x)


def map_continuous_category(
    x: Any,
    mapping: dict[str, str],
) -> str:
    """Map a continuous value to its nearest categorical prototype.

    input:
        - x: Any
        - mapping: dict[str, str]

    output:
        - category: str
    """
    if pd.isna(x) or x == "?":
        return "unknown"

    x = float(x)

    nearest = min(
        mapping.keys(),
        key=lambda k: abs(x - float(k))
    )

    return mapping[nearest]


def load_dataframe(
    config: dict[str, Any],
    filepath: str | Path | None = None,
) -> pd.DataFrame:
    """Load a configured CSV dataset into a dataframe.

    input:
        - config: dict[str, Any]
        - filepath: str | pathlib.Path | None

    output:
        - dataframe: pandas.DataFrame
    """

    filepath = resolve_configured_path(filepath or config["filepath"])

    if config.get("has_header", True):
        return pd.read_csv(filepath)

    return pd.read_csv(
        filepath,
        header=None,
        names=config["column_names"]
    )


def serialize_row(row: pd.Series, target_col: str) -> str:
    """Convert one row into a sentence-per-feature text serialization.

    input:
        - row: pandas.Series
        - target_col: str

    output:
        - serialized_row: str
    """
    sentences = []

    for col, value in row.items():
        if col == target_col:
            continue

        if pd.isna(value):
            value = "unknown"

        sentences.append(f"The {col} is {value}.")

    return " ".join(sentences)


def serialization_value(value: Any) -> Any:
    """Convert a dataframe value into a stable serializable value.

    input:
        - value: Any

    output:
        - serialized_value: Any
    """
    if pd.isna(value):
        return "unknown"

    if isinstance(value, np.generic):
        return value.item()

    return value


def serialize_row_json(row: pd.Series, target_col: str) -> str:
    """Convert one row into a one-line JSON serialization.

    input:
        - row: pandas.Series
        - target_col: str

    output:
        - serialized_row: str
    """
    row_dict = {}

    for col, value in row.items():
        if col == target_col or col.startswith("serialized_"):
            continue

        row_dict[col] = serialization_value(value)

    return json.dumps(row_dict, ensure_ascii=True)


def serialize_row_llm_style(row: pd.Series, target_col: str) -> str:
    """Convert one row into a deterministic LLM-style sentence.

    input:
        - row: pandas.Series
        - target_col: str

    output:
        - serialized_row: str
    """
    phrases = []

    for col, value in row.items():
        if col == target_col or col.startswith("serialized_"):
            continue

        phrases.append(f"{col} {serialization_value(value)}")

    return f"The patient record describes a person with {', '.join(phrases)}."


def build_llm_serialization_prompt(
    row: pd.Series,
    target_col: str,
) -> str:
    """Build a prompt for LLM-based row serialization.

    input:
        - row: pandas.Series
        - target_col: str

    output:
        - prompt: str
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


def generate_llm_serialization(
    row: pd.Series,
    target_col: str,
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
) -> str:
    """Generate one natural-language row serialization with an LLM.

    input:
        - row: pandas.Series
        - target_col: str
        - tokenizer: Any
        - model: Any
        - max_new_tokens: int

    output:
        - serialized_row: str
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


def load_llm_serializer(model_name: str) -> tuple[Any, Any]:
    """Load the optional tokenizer and model used for LLM serialization.

    input:
        - model_name: str

    output:
        - tokenizer_and_model: tuple[Any, Any]
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


def serialize_rows(
    df: pd.DataFrame,
    serializer: Callable[[pd.Series], str],
    *,
    model_name: str,
    dataset_name: str,
    serialization: str,
) -> list[str]:
    """Serialize dataframe rows while reporting periodic progress.

    input:
        - df: pandas.DataFrame
        - serializer: Callable[[pandas.Series], str]
        - model_name: str
        - dataset_name: str
        - serialization: str

    output:
        - serialized_rows: list[str]
    """
    total_rows = len(df)
    print(
        f"\nModel: {model_name} | Dataset: {dataset_name} "
        f"| Serialization: {serialization} | Rows: {total_rows}",
        flush=True,
    )
    serialized_rows = []

    for processed_rows, (_, row) in enumerate(df.iterrows(), start=1):
        serialized_rows.append(serializer(row))
        if (
            processed_rows % PROGRESS_INTERVAL == 0
            or processed_rows == total_rows
        ):
            print(
                f"  Processed {processed_rows}/{total_rows} rows",
                flush=True,
            )

    return serialized_rows


def process_dataset(
    config: dict[str, Any],
    filepath: str | Path | None = None,
    output_file: str | Path | None = None,
    serializations: Sequence[str] = SERIALIZATION_CHOICES,
    llm_tokenizer: Any = None,
    llm_model: Any = None,
    llm_max_new_tokens: int = 80,
    dataset_name: str = "unknown",
    serializer_model_name: str = "deterministic serializer",
) -> pd.DataFrame:
    """Load, clean, serialize, and save one tabular dataset split.

    input:
        - config: dict[str, Any]
        - filepath: str | pathlib.Path | None
        - output_file: str | pathlib.Path | None
        - serializations: Sequence[str]
        - llm_tokenizer: Any
        - llm_model: Any
        - llm_max_new_tokens: int
        - dataset_name: str
        - serializer_model_name: str

    output:
        - processed_dataframe: pandas.DataFrame
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
        df["serialized_text"] = serialize_rows(
            df,
            lambda row: serialize_row(row, target_col),
            model_name="deterministic text-template serializer",
            dataset_name=dataset_name,
            serialization="text_template",
        )
        output_columns.append("serialized_text")

    if "json" in serializations:
        df["serialized_json"] = serialize_rows(
            df,
            lambda row: serialize_row_json(row, target_col),
            model_name="deterministic JSON serializer",
            dataset_name=dataset_name,
            serialization="json",
        )
        output_columns.append("serialized_json")

    if "llm" in serializations:
        if llm_tokenizer is not None and llm_model is not None:
            df["serialized_llm"] = serialize_rows(
                df,
                lambda row: generate_llm_serialization(
                    row,
                    target_col,
                    llm_tokenizer,
                    llm_model,
                    llm_max_new_tokens,
                ),
                model_name=serializer_model_name,
                dataset_name=dataset_name,
                serialization="llm",
            )
        else:
            df["serialized_llm"] = serialize_rows(
                df,
                lambda row: serialize_row_llm_style(row, target_col),
                model_name="deterministic LLM-style serializer",
                dataset_name=dataset_name,
                serialization="llm",
            )
        output_columns.append("serialized_llm")

    output_columns.append(target_col)

    output_file = resolve_configured_path(output_file or config["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df[output_columns].to_csv(output_file, index=False)

    return df


def serialize(
    config_path: str | Path,
    serializations: Sequence[str] = SERIALIZATION_CHOICES,
    llm_tokenizer: Any = None,
    llm_model: Any = None,
    llm_max_new_tokens: int = 80,
    serializer_model_name: str = "deterministic serializer",
) -> dict[str, pd.DataFrame]:
    """Serialize every configured split for one dataset.

    input:
        - config_path: str | pathlib.Path
        - serializations: Sequence[str]
        - llm_tokenizer: Any
        - llm_model: Any
        - llm_max_new_tokens: int
        - serializer_model_name: str

    output:
        - serialized_splits: dict[str, pandas.DataFrame]
    """
    config = load_config(config_path)
    dataset_name = Path(config_path).stem.removesuffix("_config")

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
                dataset_name=dataset_name,
                serializer_model_name=serializer_model_name,
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
            dataset_name=f"{dataset_name}/{split}",
            serializer_model_name=serializer_model_name,
        )
        for split in input_files
    }


def run_serialization(
    datasets: Sequence[str] | None = None,
    configs: Sequence[str | Path] | None = None,
    serializations: Sequence[str] | None = None,
    llm_model: str | None = None,
    llm_max_new_tokens: int = 80,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Serialize selected datasets from Python code or notebooks.

    input:
        - datasets: Sequence[str] | None
        - configs: Sequence[str | pathlib.Path] | None
        - serializations: Sequence[str] | None
        - llm_model: str | None
        - llm_max_new_tokens: int

    output:
        - serialized_datasets: dict[str, dict[str, pandas.DataFrame]]
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
            serializer_model_name=llm_model or "deterministic serializer",
        )
        serialized_datasets[str(config_path)] = split_frames
        for split, df in split_frames.items():
            print(
                f"Serialized {config_path} [{split}]: {len(df)} rows "
                f"({', '.join(serializations)})"
            )

    return serialized_datasets


def main() -> None:
    """Parse command-line arguments and run dataset serialization.

    input:
        - None: None

    output:
        - None: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset names to serialize using their matching config files.",
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
