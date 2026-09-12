import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from .setup_config import DATASETS, RESULTS_DIR
except ImportError:  # Allow direct execution from models/tabllm.
    from setup_config import DATASETS, RESULTS_DIR


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


def load_training_times(results_dir=RESULTS_DIR):
    """
    Load run times from all_results summary files when available.
    """
    time_by_output_file = {}

    for filename in ["all_results_summary.csv", "all_results_test_summary.csv"]:
        summary_path = Path(results_dir) / filename

        if not summary_path.exists():
            continue

        summary_df = pd.read_csv(summary_path)

        if "output_file" not in summary_df.columns or "time_sec" not in summary_df.columns:
            continue

        for _, row in summary_df.iterrows():
            time_by_output_file[str(row["output_file"])] = row["time_sec"]

            if {"dataset", "serialization", "model"}.issubset(summary_df.columns):
                tuple_key = (
                    row["dataset"],
                    row["serialization"],
                    row["model"],
                )
                time_by_output_file[tuple_key] = row["time_sec"]

    return time_by_output_file


def load_metrics_json(raw_response_path):
    """
    Load metadata JSON next to a raw-response CSV when present.
    """
    metrics_path = raw_response_path.with_name(
        raw_response_path.name.replace("_raw_responses.csv", "_metrics.json")
    )

    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r") as file:
        return json.load(file)


def extract_run_info(raw_response_path, results_dir=RESULTS_DIR):
    """
    Infer dataset, serialization, model, and run tag from a raw-response path.
    """
    relative_path = raw_response_path.relative_to(results_dir)
    dataset = relative_path.parts[0]
    serialization = relative_path.parts[1]
    filename = raw_response_path.name

    if filename.endswith("_test_raw_responses.csv"):
        model = filename.replace("_test_raw_responses.csv", "")
        run_tag = "test"
    else:
        model = filename.replace("_raw_responses.csv", "")
        run_tag = None

    return dataset, serialization, model, run_tag


def evaluate_raw_responses(raw_response_path, dataset_config):
    """
    Compute metrics for one raw-response CSV.
    """
    df = pd.read_csv(raw_response_path)
    labels = dataset_config["choices"]
    positive_label = dataset_config["positive_label"]

    parsed_outputs = [
        parse_prediction(output, labels)
        for output in df["output"]
    ]
    parsed_predictions = [parsed_label for parsed_label, _ in parsed_outputs]
    parse_statuses = [status for _, status in parsed_outputs]
    y_true = df["target_col"].tolist()

    y_pred_for_scoring = parsed_predictions
    all_confusion_labels = labels

    accuracy = accuracy_score(y_true, y_pred_for_scoring)
    true_positive = sum(
        true_label == positive_label and predicted_label == positive_label
        for true_label, predicted_label in zip(y_true, y_pred_for_scoring)
    )
    false_positive = sum(
        true_label != positive_label and predicted_label == positive_label
        for true_label, predicted_label in zip(y_true, y_pred_for_scoring)
    )
    false_negative = sum(
        true_label == positive_label and predicted_label != positive_label
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
        labels=all_confusion_labels,
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
    evaluated_df["parsed_prediction"] = parsed_predictions
    evaluated_df["parse_status"] = parse_statuses
    evaluated_df["is_correct"] = evaluated_df["target_col"] == evaluated_df["parsed_prediction"]

    n_examples = len(df)
    n_non_relevant = parse_statuses.count("non_relevant")
    n_ambiguous = parse_statuses.count("ambiguous")

    metrics = {
        "n_examples": int(n_examples),
        "accuracy": accuracy,
        "positive_label": positive_label,
        "precision": positive_precision,
        "recall": positive_recall,
        "f1_score": macro_f1,
        "f1_macro": macro_f1,
        "non_relevant_pct": 100 * (n_non_relevant + n_ambiguous) / n_examples if n_examples else 0,
    }

    return metrics, confusion_df, evaluated_df


def save_confusion_matrix_image(confusion_df, output_path, title):
    """
    Save a confusion matrix dataframe as a PNG heatmap.
    """
    values = confusion_df.values
    x_labels = [
        label.replace("pred_", "")
        for label in confusion_df.columns
    ]
    y_labels = [
        label.replace("true_", "")
        for label in confusion_df.index
    ]

    fig_width = max(7, 1.8 * len(x_labels))
    fig_height = max(5.5, 1.4 * len(y_labels))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, cmap="Blues")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.set_yticklabels(y_labels)

    threshold = values.max() / 2 if values.size and values.max() > 0 else 0

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if value > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def discover_raw_response_files(results_dir=RESULTS_DIR, include_test=False):
    """
    Discover raw response CSV files under the results directory.
    """
    results_path = Path(results_dir)
    paths = sorted(results_path.glob("*/*/*_raw_responses.csv"))

    if include_test:
        return paths

    return [
        path
        for path in paths
        if not path.name.endswith("_test_raw_responses.csv")
    ]


def generate_final_metrics(results_dir=RESULTS_DIR, include_test=False):
    """
    Evaluate all raw-response files and save final metrics outputs.
    """
    results_path = Path(results_dir)
    confusion_dir = results_path / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    time_by_output_file = load_training_times(results_dir)
    rows = []

    for raw_response_path in discover_raw_response_files(results_dir, include_test=include_test):
        dataset, serialization, model, run_tag = extract_run_info(raw_response_path, results_path)

        if dataset not in DATASETS:
            print(f"Skipping {raw_response_path}: unknown dataset '{dataset}'.")
            continue

        metrics, confusion_df, _ = evaluate_raw_responses(
            raw_response_path,
            DATASETS[dataset],
        )
        metadata = load_metrics_json(raw_response_path)

        confusion_filename = f"{dataset}__{serialization}__{model}"
        if run_tag:
            confusion_filename += f"__{run_tag}"
        confusion_filename += "__confusion_matrix.png"

        confusion_path = confusion_dir / confusion_filename
        title = f"{dataset} | {serialization} | {model}"
        if run_tag:
            title += f" | {run_tag}"
        save_confusion_matrix_image(confusion_df, confusion_path, title)

        output_file_key = str(raw_response_path)
        run_key = (dataset, serialization, model)
        training_time_sec = time_by_output_file.get(
            run_key,
            time_by_output_file.get(output_file_key),
        )

        row = {
            "dataset": dataset,
            "serialization": serialization,
            "model": model,
            "confusion_matrix_file": str(confusion_path),
            "training_time_sec": training_time_sec,
            "batch_size": metadata.get("batch_size"),
            "total_examples": metadata.get("total_examples"),
        }
        row.update(metrics)
        rows.append(row)

    final_metrics = pd.DataFrame(rows)
    output_path = results_path / "final_tabular_llm_metrics.csv"
    final_metrics.to_csv(output_path, index=False)

    print(f"Saved final metrics to {output_path}")
    print(f"Saved confusion matrices to {confusion_dir}")

    return final_metrics


if __name__ == "__main__":
    generate_final_metrics()
