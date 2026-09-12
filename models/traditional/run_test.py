"""Run the traditional tabular-model benchmark outside Jupyter.

From the repository root, execute::

    venv/bin/python models/traditional/run_test.py

The script tunes XGBoost and LightGBM on each training split, evaluates the
selected models once on the held-out test splits, and writes CSV results plus
confusion matrices under ``results/xgboost`` and ``results/lightgbm``.
"""

from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.utils import (
    DATASETS,
    RESULTS_ROOT,
    compute_binary_metrics,
    load_dataset,
    predictions_from_probabilities,
    save_confusion_matrix,
)

try:
    from .lightgbm_wrapper import LightGBMEvaluator
    from .xgboost_wrapper import XGBoostEvaluator
except ImportError:  # Direct script execution.
    from lightgbm_wrapper import LightGBMEvaluator
    from xgboost_wrapper import XGBoostEvaluator


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
        predictions = predictions_from_probabilities(test_probabilities)
        test_metrics = compute_binary_metrics(
            y_test, predictions, test_probabilities, prefix=""
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
            y_test,
            predictions,
            model_name=model_name,
            dataset_name=dataset_name,
            output_path=(
                RESULTS_ROOT
                / model_name
                / f"{model_name}_{dataset_name}_CM.png"
            ),
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
