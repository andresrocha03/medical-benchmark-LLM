"""Evaluate TabICLv2 on all three preprocessed medical datasets.

Run from the repository root with::

    venv/bin/python models/tfm/iclv2/run_test.py

The first execution may download pretrained model weights through the
``tabicl`` package. Results are written under ``results/tabicl_v2``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from .wrapper import evaluate_tabicl_v2
except ImportError:  # Direct script execution.
    from wrapper import evaluate_tabicl_v2


TARGET_COLUMN = "target"
DATASETS = ("diabetes", "hepatitis", "heart")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPOSITORY_ROOT / "data" / "pre-processed"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "tabicl_v2"
RESULTS_PATH = RESULTS_DIR / "tabicl_v2_results.csv"


def load_dataset(dataset_name):
    """Load a fixed preprocessed split without creating a new test set."""
    train_data = pd.read_csv(DATA_DIR / f"{dataset_name}_train.csv")
    test_data = pd.read_csv(DATA_DIR / f"{dataset_name}_test.csv")
    if list(train_data.columns) != list(test_data.columns):
        raise ValueError(f"Train/test columns differ for {dataset_name}.")
    if TARGET_COLUMN not in train_data:
        raise ValueError(f"Missing target column {TARGET_COLUMN!r}.")

    feature_columns = [
        column for column in train_data.columns if column != TARGET_COLUMN
    ]
    X_train = train_data[feature_columns].to_numpy(dtype=np.float32)
    X_test = test_data[feature_columns].to_numpy(dtype=np.float32)
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=np.int64)

    # The shared preprocessed files already use the benchmark's 0/1 encoding,
    # so an additional LabelEncoder is neither necessary nor desirable.
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("TabICLv2 requires binary targets encoded as 0 and 1.")
    return X_train, y_train, X_test, y_test


def save_confusion_matrix(y_true, predictions, dataset_name):
    """Save one held-out confusion matrix for the current dataset."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No bad event (0)", "Bad event (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(
        f"TabICLv2 — {dataset_name.title()} test confusion matrix"
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        RESULTS_DIR / f"tabicl_v2_{dataset_name}_CM.png", dpi=150
    )
    plt.close(fig)


def test_all():
    """Run one leakage-free TabICLv2 evaluation per medical dataset."""
    results = []
    for dataset_name in DATASETS:
        print(f"\n--- {dataset_name} ---")
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        predictions, _, metrics = evaluate_tabicl_v2(
            X_train,
            y_train,
            X_test,
            y_test,
            n_estimators=8,
            device=None,
            kv_cache=True,
            random_state=42,
        )
        results.append({"dataset": dataset_name, **metrics})
        save_confusion_matrix(y_test, predictions, dataset_name)
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    test_all()
