"""Reusable TabICLv2 wrapper for binary tabular classification."""

from time import perf_counter

from models.utils import (
    compute_binary_metrics,
    predictions_from_probabilities,
    select_positive_probabilities,
)

# TabICLv2 is distributed through the ``tabicl`` package.
try:
    from tabicl import TabICLClassifier
except ImportError as error:
    raise ImportError(
        "TabICLv2 is not installed. Run: venv/bin/pip install tabicl"
    ) from error


RANDOM_STATE = 42
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
        return select_positive_probabilities(
            probabilities, self.classifier.classes_
        )

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
        predictions = predictions_from_probabilities(positive_probabilities)

        # Use the same fixed threshold and metric definitions as the other
        # benchmark pipelines. Test labels do not influence model selection.
        metrics = {
            "train_time_seconds": train_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            **compute_binary_metrics(
                y_test, predictions, positive_probabilities
            ),
        }
        return predictions, positive_probabilities, metrics


def evaluate_tabicl_v2(X_train, y_train, X_test, y_test, **model_options):
    """Convenience function matching the other foundation-model wrappers."""
    evaluator = TabICLv2Evaluator(**model_options)
    return evaluator.evaluate(X_train, y_train, X_test, y_test)
