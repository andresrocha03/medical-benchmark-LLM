"""Run the traditional tabular-model benchmark outside Jupyter.

From the repository root, execute::

    venv/bin/python models/traditional/run_test.py

The script tunes XGBoost and LightGBM on each training split, evaluates the
selected models once on the held-out test splits, and writes CSV results plus
confusion matrices under ``results/xgboost`` and ``results/lightgbm``.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .lightgbm_wrapper import LightGBMEvaluator
    from .xgboost_wrapper import XGBoostEvaluator
except ImportError:  # Direct script execution.
    from lightgbm_wrapper import LightGBMEvaluator
    from xgboost_wrapper import XGBoostEvaluator


TARGET_COLUMN = "target"
DATASETS = ("diabetes", "heart", "hepatitis")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "data" / "pre-processed"
RESULTS_ROOT = REPOSITORY_ROOT / "results"


def load_dataset(dataset_name):
    """Load one preprocessed train/test pair and separate features and target."""
    train_data = pd.read_csv(DATA_DIR / f"{dataset_name}_train.csv")
    test_data = pd.read_csv(DATA_DIR / f"{dataset_name}_test.csv")
    if list(train_data.columns) != list(test_data.columns):
        raise ValueError(f"Train/test columns differ for {dataset_name}.")
    if TARGET_COLUMN not in train_data:
        raise ValueError(f"Missing target column {TARGET_COLUMN!r}.")

    X_train = train_data.drop(columns=TARGET_COLUMN)
    y_train = train_data[TARGET_COLUMN].astype(int)
    X_test = test_data.drop(columns=TARGET_COLUMN)
    y_test = test_data[TARGET_COLUMN].astype(int)
    if set(y_train.unique()) != {0, 1} or not set(y_test.unique()) <= {0, 1}:
        raise ValueError("Traditional evaluation requires binary 0/1 targets.")
    return X_train, y_train, X_test, y_test


def calculate_test_metrics(y_true, probabilities):
    """Calculate held-out metrics with the benchmark's fixed 0.5 threshold."""
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    return predictions, {
        "f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "auc": roc_auc_score(y_true, probabilities),
    }


def save_confusion_matrix(y_true, predictions, model_name, dataset_name):
    """Save the held-out confusion matrix for one model and dataset."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No bad event (0)", "Bad event (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{model_name} — {dataset_name.title()} test confusion matrix")
    fig.tight_layout()
    model_results_dir = RESULTS_ROOT / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        model_results_dir / f"{model_name}_{dataset_name}_CM.png", dpi=150
    )
    plt.close(fig)


def evaluate_model(evaluator_class):
    """Tune and test one model family on all configured datasets."""
    model_name = evaluator_class.model_name
    model_results = []

    for dataset_name in DATASETS:
        print(f"\n--- {model_name}: {dataset_name} ---")
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        evaluator = evaluator_class()

        # Training time includes every cross-validation fit plus GridSearchCV's
        # final refit of the best configuration on the complete training split.
        training_start = perf_counter()
        evaluator.fit(X_train, y_train)
        train_time_seconds = perf_counter() - training_start

        # Prediction time covers held-out probability inference only.
        prediction_start = perf_counter()
        test_probabilities = evaluator.predict_proba(X_test)
        prediction_time_seconds = perf_counter() - prediction_start
        predictions, test_metrics = calculate_test_metrics(
            y_test, test_probabilities
        )
        validation_metrics = evaluator.cross_validation_metrics()

        row = {
            "dataset": dataset_name,
            "train_time_seconds": train_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            "best_parameters": str(evaluator.best_parameters),
            **{
                f"train_{metric}": value
                for metric, value in validation_metrics.items()
            },
            **{
                f"test_{metric}": value
                for metric, value in test_metrics.items()
            },
        }
        model_results.append(row)
        save_confusion_matrix(
            y_test, predictions, model_name, dataset_name
        )
        for metric, value in row.items():
            if metric != "dataset":
                print(f"  {metric}: {value}")

    model_results_dir = RESULTS_ROOT / model_name
    model_results_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_results_dir / f"{model_name}_results.csv"
    pd.DataFrame(model_results).to_csv(results_path, index=False)
    print(f"\nSaved {results_path.relative_to(REPOSITORY_ROOT)}")


def test_all():
    """Evaluate XGBoost first, followed by LightGBM."""
    for evaluator_class in (XGBoostEvaluator, LightGBMEvaluator):
        evaluate_model(evaluator_class)


if __name__ == "__main__":
    test_all()
