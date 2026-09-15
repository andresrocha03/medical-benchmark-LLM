"""Reusable TabICLv2 wrapper for binary tabular classification."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

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
        n_estimators: int = 8,
        device: str | None = None,
        kv_cache: bool = True,
        random_state: int = RANDOM_STATE,
        checkpoint_version: str = CHECKPOINT_VERSION,
    ) -> None:
        """Configure a pretrained TabICLv2 classifier.

        input:
            - n_estimators: int
            - device: str | None
            - kv_cache: bool
            - random_state: int
            - checkpoint_version: str

        output:
            - None: None
        """
        self.classifier = TabICLClassifier(
            n_estimators=n_estimators,
            device=device,
            kv_cache=kv_cache,
            random_state=random_state,
            # Pin the published v2 classifier rather than relying on whichever
            # checkpoint a future package release chooses as its default.
            checkpoint_version=checkpoint_version,
        )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
    ) -> TabICLv2Evaluator:
        """Fit the classifier on the supplied training context.

        input:
            - X_train: pandas.DataFrame | numpy.ndarray
            - y_train: numpy.ndarray

        output:
            - evaluator: TabICLv2Evaluator
        """
        self.classifier.fit(X_train, y_train)
        return self

    def predict_positive_probability(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return probabilities assigned to the positive class.

        input:
            - X: pandas.DataFrame | numpy.ndarray

        output:
            - positive_probabilities: numpy.ndarray
        """
        probabilities = self.classifier.predict_proba(X)
        return select_positive_probabilities(
            probabilities, self.classifier.classes_
        )

    def evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Fit the classifier and compute held-out benchmark metrics.

        input:
            - X_train: pandas.DataFrame | numpy.ndarray
            - y_train: numpy.ndarray
            - X_test: pandas.DataFrame | numpy.ndarray
            - y_test: numpy.ndarray

        output:
            - evaluation: tuple[numpy.ndarray, numpy.ndarray, dict[str, float]]
        """
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


def evaluate_tabicl_v2(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
    **model_options: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Evaluate TabICLv2 with optional classifier configuration.

    input:
        - X_train: pandas.DataFrame | numpy.ndarray
        - y_train: numpy.ndarray
        - X_test: pandas.DataFrame | numpy.ndarray
        - y_test: numpy.ndarray
        - model_options: object

    output:
        - evaluation: tuple[numpy.ndarray, numpy.ndarray, dict[str, float]]
    """
    evaluator = TabICLv2Evaluator(**model_options)
    return evaluator.evaluate(X_train, y_train, X_test, y_test)
