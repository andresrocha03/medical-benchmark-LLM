"""Shared data, evaluation, plotting, and reproducibility utilities."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 42
POSITIVE_THRESHOLD = 0.5
TARGET_COLUMN = "target"
DATASETS = ("diabetes", "hepatitis", "heart")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "data" / "pre-processed"
RESULTS_ROOT = REPOSITORY_ROOT / "results"
PROGRESS_INTERVAL = 150

CROSS_VALIDATION_SCORING = {
    "f1_macro": make_scorer(
        f1_score, average="macro", zero_division=0
    ),
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "auc": "roc_auc",
}


def print_run_context(model_name, dataset_name, *, total_rows=None):
    """Print the model and dataset at the start of a benchmark run."""
    message = f"Model: {model_name} | Dataset: {dataset_name}"
    if total_rows is not None:
        message += f" | Rows: {total_rows}"
    print(f"\n{message}", flush=True)


def print_row_progress(processed_rows, total_rows, *, interval=PROGRESS_INTERVAL):
    """Print a heartbeat at each row interval and when processing finishes."""
    if processed_rows % interval == 0 or processed_rows == total_rows:
        print(
            f"  Processed {processed_rows}/{total_rows} rows",
            flush=True,
        )


def load_dataset(dataset_name, *, as_numpy=False):
    """Load and validate one fixed, preprocessed medical train/test split."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; expected one of {DATASETS}."
        )

    train_data = pd.read_csv(DATA_DIR / f"{dataset_name}_train.csv")
    test_data = pd.read_csv(DATA_DIR / f"{dataset_name}_test.csv")
    if list(train_data.columns) != list(test_data.columns):
        raise ValueError(f"Train/test columns differ for {dataset_name}.")
    if TARGET_COLUMN not in train_data:
        raise ValueError(f"Missing target column {TARGET_COLUMN!r}.")

    X_train = train_data.drop(columns=TARGET_COLUMN)
    X_test = test_data.drop(columns=TARGET_COLUMN)
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if set(np.unique(y_train)) != {0, 1} or not set(np.unique(y_test)) <= {0, 1}:
        raise ValueError("Evaluation requires binary targets encoded as 0 and 1.")

    if as_numpy:
        X_train = X_train.to_numpy(dtype=np.float32)
        X_test = X_test.to_numpy(dtype=np.float32)
    return X_train, y_train, X_test, y_test


def predictions_from_probabilities(
    positive_probabilities, threshold=POSITIVE_THRESHOLD
):
    """Convert positive-class probabilities to binary predictions."""
    return (np.asarray(positive_probabilities) >= threshold).astype(np.int64)


def select_positive_probabilities(probabilities, classes):
    """Select class-1 probabilities from a binary classifier output."""
    probabilities = np.asarray(probabilities)
    positive_class_indices = np.flatnonzero(np.asarray(classes) == 1)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Evaluation requires probabilities for two classes.")
    if len(positive_class_indices) != 1:
        raise ValueError("Evaluation requires a unique positive class labeled 1.")
    return probabilities[:, positive_class_indices[0]]


def compute_binary_metrics(
    y_true,
    predictions,
    positive_probabilities,
    *,
    prefix="test_",
):
    """Compute the benchmark's common held-out binary metrics."""
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)
    positive_probabilities = np.asarray(positive_probabilities)
    auc = (
        roc_auc_score(y_true, positive_probabilities)
        if len(np.unique(y_true)) == 2
        else np.nan
    )
    return {
        f"{prefix}f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        f"{prefix}accuracy": accuracy_score(y_true, predictions),
        f"{prefix}precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        f"{prefix}recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        f"{prefix}auc": auc,
    }


def save_confusion_matrix(
    y_true,
    predictions,
    *,
    model_name,
    dataset_name,
    output_path,
    labels=None,
    display_labels=None,
    title=None,
):
    """Save one consistently styled held-out confusion matrix."""
    if labels is None:
        labels = [0, 1]
    if display_labels is None:
        display_labels = (
            ["No bad event (0)", "Bad event (1)"]
            if list(labels) == [0, 1]
            else [str(label) for label in labels]
        )

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=labels,
        display_labels=display_labels,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(
        title
        or f"{model_name} — {dataset_name.title()} test confusion matrix"
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def set_random_seed(seed=SEED):
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_torch_device():
    """Return the CUDA device name when available, otherwise CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
