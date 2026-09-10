"""Evaluate frozen YAGO-pretrained TARTE + XGBoost and save results."""

import argparse
import json
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
    from .common import SEED, get_device, set_seed
    from .pretrained import DEFAULT_CHECKPOINT, FastTextEncoder, RowEncoder, load_backbone
    from .tarte_featurizer_wrapper import TARTEFeaturizerEvaluator
except ImportError:
    from common import SEED, get_device, set_seed
    from pretrained import DEFAULT_CHECKPOINT, FastTextEncoder, RowEncoder, load_backbone
    from tarte_featurizer_wrapper import TARTEFeaturizerEvaluator


TARGET_COLUMN = "target"
DATASETS = ("diabetes", "hepatitis", "heart")

# Resolve paths from this file so execution works from any current directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPOSITORY_ROOT / "data" / "pre-processed"
RESULTS_DIR = REPOSITORY_ROOT / "results" / "tarte"
RESULTS_PATH = RESULTS_DIR / "tarte_results.csv"


def load_dataset(dataset_name):
    """Load one fixed train/test split and validate the benchmark contract."""
    train_data = pd.read_csv(DATA_DIR / f"{dataset_name}_train.csv")
    test_data = pd.read_csv(DATA_DIR / f"{dataset_name}_test.csv")
    if list(train_data.columns) != list(test_data.columns):
        raise ValueError(f"Train/test columns differ for {dataset_name}.")
    if TARGET_COLUMN not in train_data:
        raise ValueError(f"Missing target column {TARGET_COLUMN!r}.")

    # Keep column names for their language-model-style embeddings, but pass
    # float32 arrays to PyTorch and integer targets to the classifiers.
    X_train_frame = train_data.drop(columns=TARGET_COLUMN)
    X_test_frame = test_data.drop(columns=TARGET_COLUMN)
    X_train = X_train_frame.to_numpy(dtype=np.float32)
    X_test = X_test_frame.to_numpy(dtype=np.float32)
    y_train = train_data[TARGET_COLUMN].to_numpy(dtype=np.int64)
    y_test = test_data[TARGET_COLUMN].to_numpy(dtype=np.int64)
    if set(np.unique(y_train)) != {0, 1} or not set(np.unique(y_test)) <= {0, 1}:
        raise ValueError("TARTE evaluation requires binary targets encoded as 0/1.")
    return X_train, y_train, X_test, y_test, list(X_train_frame.columns)


def compute_metrics(
    y_true,
    predictions,
    positive_probabilities,
    train_time_seconds,
    prediction_time_seconds,
):
    """Compute the same metrics and timing fields as the other model families."""
    return {
        "train_time_seconds": train_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "test_f1_macro": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "test_accuracy": accuracy_score(y_true, predictions),
        "test_precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "test_recall": recall_score(y_true, predictions, zero_division=0),
        "test_auc": roc_auc_score(y_true, positive_probabilities) if len(np.unique(y_true)) == 2 else np.nan,
    }


def save_confusion_matrix(y_true, predictions, dataset_name, approach_name):
    """Save one consistently styled test confusion matrix per experiment."""
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
        f"TARTE-{approach_name} — {dataset_name.title()} test confusion matrix"
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"tarte_{approach_name.lower()}_{dataset_name}_CM.png"
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def test_all(fasttext_path):
    """Load a saved backbone once; fit only XGBoost on each medical train split."""
    if not DEFAULT_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {DEFAULT_CHECKPOINT}. Run pretrain_tarte.py first.")
    set_seed(SEED)
    device = get_device()
    text = FastTextEncoder(fasttext_path)
    backbone, checkpoint = load_backbone(DEFAULT_CHECKPOINT, text, device)
    print(f"Device: {device}; checkpoint: {DEFAULT_CHECKPOINT}")
    results = []
    for dataset_name in DATASETS:
        X_train, y_train, X_test, y_test, names = load_dataset(dataset_name)
        train_rows = [list(zip(names, row)) for row in X_train]
        test_rows = [list(zip(names, row)) for row in X_test]
        started = perf_counter()
        encoder = RowEncoder(text).fit(train_rows)
        encoded_train = [encoder.encode(row) for row in train_rows]
        evaluator = TARTEFeaturizerEvaluator(backbone, device)
        evaluator.fit(encoded_train, y_train)
        train_time = perf_counter() - started
        started = perf_counter()
        encoded_test = [encoder.encode(row) for row in test_rows]
        probabilities = evaluator.predict_proba(encoded_test)
        prediction_time = perf_counter() - started
        predictions = (probabilities >= 0.5).astype(int)
        metrics = compute_metrics(y_test, predictions, probabilities, train_time, prediction_time)
        results.append({"dataset": dataset_name, "approach": "Featurizer-XGBoost",
                        "checkpoint": str(DEFAULT_CHECKPOINT.resolve()),
                        "pretraining_steps": checkpoint["metadata"]["steps"],
                        "pretraining_rows": checkpoint["metadata"]["sample_rows"], **metrics})
        save_confusion_matrix(y_test, predictions, dataset_name, "Featurizer-XGBoost")
        print(f"{dataset_name}: {metrics}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    (RESULTS_DIR / "tarte_run_metadata.json").write_text(json.dumps({
        "checkpoint": str(DEFAULT_CHECKPOINT.resolve()), "config": checkpoint["config"],
        "pretraining": checkpoint["metadata"], "evaluation_fasttext": text.path,
        "seed": SEED, "datasets": list(DATASETS),
        "xgboost_parameters": evaluator.classifier.get_params(),
    }, indent=2) + "\n")
    print(f"Saved {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasttext-model", type=Path, required=True)
    args = parser.parse_args()
    test_all(args.fasttext_model)


if __name__ == "__main__":
    main()
