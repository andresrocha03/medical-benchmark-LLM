import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from ..utils import save_confusion_matrix
except ImportError:  # Allow direct execution from models/tabllm.
    import sys

    REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from models.utils import save_confusion_matrix

try:
    from .config.setup_config import DATASETS, PREDICTION_LABELS, RESULTS_DIR
except ImportError:  # Allow direct execution from models/tabllm.
    from models.tabllm.config.setup_config import (
        DATASETS,
        PREDICTION_LABELS,
        RESULTS_DIR,
    )


INVALID_LABEL = "__invalid__"


def normalize_text(text):
    """
    Normalize model output and labels for robust matching.
    """
    text = "" if pd.isna(text) else str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_label_mentions(output, labels):
    """
    Find valid label mentions in a noisy model output.

    Longer labels are matched first so labels such as
    "no heart disease" are not double-counted as "heart disease".
    """
    normalized_output = normalize_text(output)
    mentions = []

    for label in sorted(labels, key=len, reverse=True):
        normalized_label = normalize_text(label)
        pattern = rf"(?<!\w){re.escape(normalized_label)}(?!\w)"

        for match in re.finditer(pattern, normalized_output):
            start, end = match.span()

            if any(start >= old_start and end <= old_end for old_start, old_end, _ in mentions):
                continue

            mentions.append((start, end, label))

    mentions.sort(key=lambda item: item[0])
    return mentions


def parse_prediction(output, labels):
    """
    Parse one model output into a valid label when possible.

    Returns
    -------
    tuple
        (parsed_label, status), where status is one of:
        - "parsed": exactly one valid label was found.
        - "non_relevant": no valid label was found.
        - "ambiguous": more than one distinct valid label was found.
    """
    mentions = find_label_mentions(output, labels)
    mentioned_labels = []

    for _, _, label in mentions:
        if label not in mentioned_labels:
            mentioned_labels.append(label)

    if len(mentioned_labels) == 1:
        return mentioned_labels[0], "parsed"

    if len(mentioned_labels) == 0:
        return INVALID_LABEL, "non_relevant"

    return INVALID_LABEL, "ambiguous"


def extract_run_info(raw_response_path, results_dir=RESULTS_DIR):
    """
    Infer dataset, serialization, and model from a raw-response path.
    """
    relative_path = raw_response_path.relative_to(results_dir)
    dataset = relative_path.parts[0]
    serialization = relative_path.parts[1]
    filename = raw_response_path.name

    model = filename.replace("_raw_responses.csv", "")

    return dataset, serialization, model


def evaluate_raw_responses(raw_response_path, dataset_config):
    """
    Compute metrics for one raw-response CSV.
    """
    df = pd.read_csv(raw_response_path)
    target_labels = dataset_config["choices"]
    positive_target_label = dataset_config["positive_label"]
    negative_target_labels = [
        label for label in target_labels if label != positive_target_label
    ]
    if len(target_labels) != 2 or len(negative_target_labels) != 1:
        raise ValueError(
            "TabLLM requires exactly two target labels and one positive label."
        )

    negative_response, positive_response = PREDICTION_LABELS
    target_to_response = {
        positive_target_label: positive_response,
        negative_target_labels[0]: negative_response,
    }

    parsed_outputs = [
        parse_prediction(output, PREDICTION_LABELS)
        for output in df["output"]
    ]
    parsed_predictions = [parsed_label for parsed_label, _ in parsed_outputs]
    parse_statuses = [status for _, status in parsed_outputs]
    y_true = df["target_col"].map(target_to_response)
    if y_true.isna().any():
        unknown_targets = sorted(df.loc[y_true.isna(), "target_col"].unique())
        raise ValueError(f"Unknown target labels in raw responses: {unknown_targets}")
    y_true = y_true.tolist()

    y_pred_for_scoring = parsed_predictions
    # Include invalid model answers in the matrix instead of silently dropping
    # them from its column totals.
    all_confusion_labels = [*PREDICTION_LABELS, INVALID_LABEL]

    accuracy = accuracy_score(y_true, y_pred_for_scoring)
    true_positive = sum(
        true_label == positive_response and predicted_label == positive_response
        for true_label, predicted_label in zip(y_true, y_pred_for_scoring)
    )
    false_positive = sum(
        true_label != positive_response and predicted_label == positive_response
        for true_label, predicted_label in zip(y_true, y_pred_for_scoring)
    )
    false_negative = sum(
        true_label == positive_response and predicted_label != positive_response
        for true_label, predicted_label in zip(y_true, y_pred_for_scoring)
    )
    positive_precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0
    )
    positive_recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0
    )
    macro_f1 = f1_score(
        y_true,
        y_pred_for_scoring,
        labels=PREDICTION_LABELS,
        average="macro",
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred_for_scoring,
        labels=all_confusion_labels,
    )
    confusion_df = pd.DataFrame(
        confusion,
        index=[f"true_{label}" for label in all_confusion_labels],
        columns=[f"pred_{label}" for label in all_confusion_labels],
    )

    evaluated_df = df.copy()
    evaluated_df["expected_prediction"] = y_true
    evaluated_df["parsed_prediction"] = parsed_predictions
    evaluated_df["parse_status"] = parse_statuses
    evaluated_df["is_correct"] = (
        evaluated_df["expected_prediction"]
        == evaluated_df["parsed_prediction"]
    )

    n_examples = len(df)
    n_non_relevant = parse_statuses.count("non_relevant")
    n_ambiguous = parse_statuses.count("ambiguous")

    metrics = {
        "n_examples": int(n_examples),
        "test_accuracy": accuracy,
        "positive_label": positive_response,
        "positive_target_label": positive_target_label,
        "test_precision": positive_precision,
        "test_recall": positive_recall,
        "test_f1_macro": macro_f1,
        # AUC requires continuous scores or probabilities. TabLLM currently
        # produces only hard text labels, so a comparable AUC is unavailable.
        "test_auc": float("nan"),
        "invalid_response_pct": (
            100 * (n_non_relevant + n_ambiguous) / n_examples
            if n_examples
            else 0
        ),
        "non_relevant_responses": int(n_non_relevant),
        "ambiguous_responses": int(n_ambiguous),
    }

    return metrics, confusion_df, evaluated_df


def discover_raw_response_files(results_dir=RESULTS_DIR):
    """
    Discover raw response CSV files under the results directory.
    """
    results_path = Path(results_dir)
    return sorted(results_path.glob("*/*/*_raw_responses.csv"))


def evaluate_and_save_run(
    raw_response_path,
    *,
    dataset,
    serialization,
    model,
    prediction_time_seconds=None,
    total_examples=None,
    batch_size=None,
    results_dir=RESULTS_DIR,
):
    """Evaluate one model/serialization/dataset trio and save its matrix."""
    raw_response_path = Path(raw_response_path)
    metrics, _, evaluated_df = evaluate_raw_responses(
        raw_response_path,
        DATASETS[dataset],
    )

    matrix_path = (
        Path(results_dir) / dataset / serialization / f"{model}_CM.png"
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"{model} | {serialization} | {dataset}"
    labels = [*PREDICTION_LABELS, INVALID_LABEL]
    save_confusion_matrix(
        evaluated_df["expected_prediction"],
        evaluated_df["parsed_prediction"],
        model_name=model,
        dataset_name=dataset,
        output_path=matrix_path,
        labels=labels,
        display_labels=[*PREDICTION_LABELS, "invalid response"],
        title=title,
    )

    return {
        "dataset": dataset,
        "model": model,
        "serialization": serialization,
        # Zero-shot TabLLM evaluation has no fitting/training phase.
        "train_time_seconds": 0.0,
        "prediction_time_seconds": prediction_time_seconds,
        "batch_size": batch_size,
        "total_examples": total_examples,
        **metrics,
        "confusion_matrix_file": str(matrix_path),
        "raw_responses_file": str(raw_response_path),
    }


def generate_final_metrics(results_dir=RESULTS_DIR):
    """
    Rebuild the trio-level results CSV from saved raw responses.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    output_path = results_path / "tabllm_results.csv"
    existing_by_trio = {}

    if output_path.exists():
        existing_results = pd.read_csv(output_path)
        required_columns = {"dataset", "serialization", "model"}
        if required_columns.issubset(existing_results.columns):
            existing_by_trio = {
                (row.dataset, row.serialization, row.model): row
                for row in existing_results.itertuples(index=False)
            }

    rows = []

    for raw_response_path in discover_raw_response_files(results_dir):
        dataset, serialization, model = extract_run_info(
            raw_response_path,
            results_path,
        )

        if dataset not in DATASETS:
            print(f"Skipping {raw_response_path}: unknown dataset '{dataset}'.")
            continue

        previous = existing_by_trio.get((dataset, serialization, model))
        rows.append(
            evaluate_and_save_run(
                raw_response_path,
                dataset=dataset,
                serialization=serialization,
                model=model,
                prediction_time_seconds=(
                    getattr(previous, "prediction_time_seconds", None)
                    if previous is not None
                    else None
                ),
                total_examples=(
                    getattr(previous, "total_examples", None)
                    if previous is not None
                    else None
                ),
                batch_size=(
                    getattr(previous, "batch_size", None)
                    if previous is not None
                    else None
                ),
                results_dir=results_path,
            )
        )

    final_metrics = pd.DataFrame(rows)
    if not final_metrics.empty:
        final_metrics = final_metrics.sort_values(
            ["dataset", "model", "serialization"]
        )
    final_metrics.to_csv(output_path, index=False)

    print(f"Saved TabLLM results to {output_path}")

    return final_metrics


if __name__ == "__main__":
    generate_final_metrics()
