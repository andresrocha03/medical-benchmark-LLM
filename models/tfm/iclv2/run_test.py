"""Evaluate TabICLv2 on all three preprocessed medical datasets.

Run from the repository root with::

    venv/bin/python models/tfm/iclv2/run_test.py

The first execution may download pretrained model weights through the
``tabicl`` package. Results are written under ``results/tabicl_v2``.
"""

from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.utils import (
    DATASETS,
    RESULTS_ROOT,
    load_dataset,
    save_confusion_matrix,
)

try:
    from .wrapper import evaluate_tabicl_v2
except ImportError:  # Direct script execution.
    from wrapper import evaluate_tabicl_v2


RESULTS_DIR = RESULTS_ROOT / "tabicl_v2"
RESULTS_PATH = RESULTS_DIR / "tabicl_v2_results.csv"


def test_all():
    """Run one leakage-free TabICLv2 evaluation per medical dataset."""
    results = []
    for dataset_name in DATASETS:
        print(f"\n--- {dataset_name} ---")
        X_train, y_train, X_test, y_test = load_dataset(
            dataset_name, as_numpy=True
        )
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
        save_confusion_matrix(
            y_test,
            predictions,
            model_name="TabICLv2",
            dataset_name=dataset_name,
            output_path=RESULTS_DIR / f"tabicl_v2_{dataset_name}_CM.png",
        )
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    test_all()
