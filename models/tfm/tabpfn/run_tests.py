"""
Standalone TabPFN evaluation script.

Run with `python models/tfm/tabpfn/run_tests.py` from the repository root.
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
    from .wrapper import evaluate_tabpfn
except ImportError:  # Direct script execution.
    from wrapper import evaluate_tabpfn


RESULTS_DIR = RESULTS_ROOT / "tabpfn"
RESULTS_PATH = RESULTS_DIR / "tabpfn_results.csv"


def test_all():
    results = []
    for dataset in DATASETS:
        print(f"\n--- {dataset} ---")
        X_train, y_train, X_test, y_test = load_dataset(dataset)
        predictions, metrics = evaluate_tabpfn(X_train, y_train, X_test, y_test)
        results.append({"dataset": dataset, **metrics})
        save_confusion_matrix(
            y_test,
            predictions,
            model_name="TabPFN",
            dataset_name=dataset,
            output_path=RESULTS_DIR / f"tabpfn_{dataset}_CM.png",
        )
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    test_all()
