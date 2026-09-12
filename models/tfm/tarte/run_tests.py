"""Evaluate the official frozen TARTE featurizer with XGBoost."""

import json
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.utils import (
    DATASETS,
    RESULTS_ROOT,
    SEED,
    compute_binary_metrics,
    get_torch_device,
    load_dataset,
    save_confusion_matrix,
    set_random_seed,
)

try:
    from .tarte_featurizer_wrapper import TARTEFeaturizerEvaluator
except ImportError:
    from tarte_featurizer_wrapper import TARTEFeaturizerEvaluator


RESULTS_DIR = RESULTS_ROOT / "tarte"
RESULTS_PATH = RESULTS_DIR / "tarte_results.csv"


def test_all():
    """Run the authors' frozen TARTE featurizer on every medical split."""
    set_random_seed(SEED)
    device = get_torch_device()
    tarte_version = version("tarte-ai")
    print(f"Device: {device}; tarte-ai: {tarte_version}")
    results = []
    for dataset_name in DATASETS:
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        started = perf_counter()
        evaluator = TARTEFeaturizerEvaluator(device=device, layer_index=2)
        evaluator.fit(X_train, y_train)
        train_time = perf_counter() - started
        started = perf_counter()
        predictions, probabilities = evaluator.predict_with_probabilities(X_test)
        prediction_time = perf_counter() - started
        metrics = {
            "train_time_seconds": train_time,
            "prediction_time_seconds": prediction_time,
            **compute_binary_metrics(y_test, predictions, probabilities),
        }
        results.append(
            {
                "dataset": dataset_name,
                "approach": "Featurizer-XGBoost",
                "tarte_ai_version": tarte_version,
                "layer_index": 2,
                **metrics,
            }
        )
        save_confusion_matrix(
            y_test,
            predictions,
            model_name="TARTE-Featurizer-XGBoost",
            dataset_name=dataset_name,
            output_path=(
                RESULTS_DIR
                / f"tarte_featurizer-xgboost_{dataset_name}_CM.png"
            ),
        )
        print(f"{dataset_name}: {metrics}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    (RESULTS_DIR / "tarte_run_metadata.json").write_text(
        json.dumps(
            {
                "tarte_ai_version": tarte_version,
                "layer_index": 2,
                "device": device,
                "seed": SEED,
                "datasets": list(DATASETS),
                "xgboost_parameters": evaluator.classifier.get_params(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {RESULTS_PATH}")


def main():
    test_all()


if __name__ == "__main__":
    main()
