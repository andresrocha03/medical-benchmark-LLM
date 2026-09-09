"""
Standalone TabPFN evaluation script.

Run with `python models/tfm/tabpfn/run_tests.py` from the repository root.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from .wrapper import evaluate_tabpfn
except ImportError:  # Direct script execution.
    from wrapper import evaluate_tabpfn


TARGET_COLUMN = "target"
DATASETS = ("diabetes", "hepatitis", "heart")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPOSITORY_ROOT / "data" / "pre-processed"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "tabpfn"
RESULTS_PATH = RESULTS_DIR / "tabpfn_results.csv"


def load_dataset(dataset_name):
    train_data = pd.read_csv(DATA_DIR / f"{dataset_name}_train.csv")
    test_data = pd.read_csv(DATA_DIR / f"{dataset_name}_test.csv")

    X_train = train_data.drop(columns=TARGET_COLUMN)
    y_train = train_data[TARGET_COLUMN].astype(int)
    X_test = test_data.drop(columns=TARGET_COLUMN)
    y_test = test_data[TARGET_COLUMN].astype(int)
    return X_train, y_train, X_test, y_test


def save_confusion_matrix(y_true, predictions, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No bad event (0)", "Bad event (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"TabPFN — {dataset_name.title()} test confusion matrix")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS_DIR / f"tabpfn_{dataset_name}_CM.png", dpi=150)
    plt.close(fig)


def test_all():
    results = []
    for dataset in DATASETS:
        print(f"\n--- {dataset} ---")
        X_train, y_train, X_test, y_test = load_dataset(dataset)
        predictions, metrics = evaluate_tabpfn(X_train, y_train, X_test, y_test)
        results.append({"dataset": dataset, **metrics})
        save_confusion_matrix(y_test, predictions, dataset)
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {RESULTS_PATH.relative_to(REPOSITORY_ROOT)}")


if __name__ == "__main__":
    test_all()
