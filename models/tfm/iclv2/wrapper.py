"""Reusable TabICLv2 wrapper for binary tabular classification."""

from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# TabICLv2 is distributed through the ``tabicl`` package.
try:
    from tabicl import TabICLClassifier
except ImportError as error:
    raise ImportError(
        "TabICLv2 is not installed. Run: venv/bin/pip install tabicl"
    ) from error


RANDOM_STATE = 42
POSITIVE_THRESHOLD = 0.5
CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"


class TabICLv2Evaluator:
    """Fit and evaluate the pretrained TabICLv2 classifier.

    TabICLv2 performs in-context learning. Its ``fit`` method primarily stores
    the training context, while most computation happens during prediction.
    The separate timers below make that behavior visible in benchmark results.
    """

    def __init__(
        self,
        n_estimators=8,
        device=None,
        kv_cache=True,
        random_state=RANDOM_STATE,
        checkpoint_version=CHECKPOINT_VERSION,
    ):
        self.classifier = TabICLClassifier(
            n_estimators=n_estimators,
            device=device,
            kv_cache=kv_cache,
            random_state=random_state,
            # Pin the published v2 classifier rather than relying on whichever
            # checkpoint a future package release chooses as its default.
            checkpoint_version=checkpoint_version,
        )

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)
        return self

    def predict_positive_probability(self, X):
        """Return the probability assigned to the benchmark's positive class."""
        probabilities = self.classifier.predict_proba(X)
        positive_class_indices = np.flatnonzero(self.classifier.classes_ == 1)
        if probabilities.shape[1] != 2 or len(positive_class_indices) != 1:
            raise ValueError(
                "TabICLv2 evaluation requires binary labels with positive class 1."
            )
        return probabilities[:, positive_class_indices[0]]

    def evaluate(self, X_train, y_train, X_test, y_test):
        """Fit on training data and compute held-out benchmark metrics."""
        training_start = perf_counter()
        self.fit(X_train, y_train)
        train_time_seconds = perf_counter() - training_start

        # Call predict_proba only once. Calling predict separately would repeat
        # TabICLv2's expensive in-context inference step.
        prediction_start = perf_counter()
        positive_probabilities = self.predict_positive_probability(X_test)
        prediction_time_seconds = perf_counter() - prediction_start
        predictions = (
            positive_probabilities >= POSITIVE_THRESHOLD
        ).astype(int)

        # Use the same fixed threshold and metric definitions as the other
        # benchmark pipelines. Test labels do not influence model selection.
        metrics = {
            "train_time_seconds": train_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            "test_f1_macro": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "test_auc": roc_auc_score(y_test, positive_probabilities),
        }
        return predictions, positive_probabilities, metrics


def evaluate_tabicl_v2(X_train, y_train, X_test, y_test, **model_options):
    """Convenience function matching the other foundation-model wrappers."""
    evaluator = TabICLv2Evaluator(**model_options)
    return evaluator.evaluate(X_train, y_train, X_test, y_test)
