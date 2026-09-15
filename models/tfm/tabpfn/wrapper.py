import math
from time import perf_counter

import numpy as np
import pandas as pd

from models.utils import (
    PROGRESS_INTERVAL,
    compute_binary_metrics,
    predictions_from_probabilities,
    print_row_progress,
    select_positive_probabilities,
)

# pyrefly: ignore [missing-import]
from tabpfn import TabPFNClassifier


class TabPFNEvaluator:
    def __init__(self, batch_size: int = PROGRESS_INTERVAL) -> None:
        """Configure a cached TabPFN classifier and prediction batch size.

        input:
            - batch_size: int

        output:
            - None: None
        """
        self.classifier = TabPFNClassifier(fit_mode="fit_with_cache")
        self.batch_size = batch_size
        
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Fit TabPFN on the training data.

        input:
            - X_train: pandas.DataFrame | numpy.ndarray
            - y_train: numpy.ndarray

        output:
            - None: None
        """
        self.classifier.fit(X_train, y_train)
        
    def _batched_predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities in memory-bounded batches.

        input:
            - X: pandas.DataFrame | numpy.ndarray

        output:
            - probabilities: numpy.ndarray
        """
        n_samples = len(X)
        n_batches = math.ceil(n_samples / self.batch_size)
        probs = []
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]
            
            batch_probs = self.classifier.predict_proba(batch_X)
            probs.append(batch_probs)
            print_row_progress(end_idx, n_samples)
            
        return np.vstack(probs)

    def evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Fit TabPFN and compute held-out benchmark metrics.

        input:
            - X_train: pandas.DataFrame | numpy.ndarray
            - y_train: numpy.ndarray
            - X_test: pandas.DataFrame | numpy.ndarray
            - y_test: numpy.ndarray

        output:
            - evaluation: tuple[numpy.ndarray, dict[str, float]]
        """
        training_start_time = perf_counter()
        self.fit(X_train, y_train)
        train_time_seconds = perf_counter() - training_start_time

        prediction_start_time = perf_counter()
        probs = self._batched_predict_proba(X_test)
        prediction_time_seconds = perf_counter() - prediction_start_time

        positive_probabilities = select_positive_probabilities(
            probs, self.classifier.classes_
        )
        predictions = predictions_from_probabilities(positive_probabilities)
        metrics = {
            "train_time_seconds": train_time_seconds,
            "prediction_time_seconds": prediction_time_seconds,
            **compute_binary_metrics(
                y_test, predictions, positive_probabilities
            ),
        }
        return predictions, metrics


def evaluate_tabpfn(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate TabPFN using the default wrapper configuration.

    input:
        - X_train: pandas.DataFrame | numpy.ndarray
        - y_train: numpy.ndarray
        - X_test: pandas.DataFrame | numpy.ndarray
        - y_test: numpy.ndarray

    output:
        - evaluation: tuple[numpy.ndarray, dict[str, float]]
    """
    evaluator = TabPFNEvaluator()
    return evaluator.evaluate(X_train, y_train, X_test, y_test)
