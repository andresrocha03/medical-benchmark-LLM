"""XGBoost classification on features from the official frozen TARTE model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from tarte_ai import TARTE_TableEncoder, TARTE_TablePreprocessor
from xgboost import XGBClassifier


class TARTEFeaturizerEvaluator:
    """Official frozen TARTE featurizer followed by XGBoost."""

    def __init__(
        self,
        device: str = "cpu",
        layer_index: int = 2,
    ) -> None:
        """Configure the TARTE feature encoder and XGBoost classifier.

        input:
            - device: str
            - layer_index: int

        output:
            - None: None
        """
        self.pipeline = Pipeline(
            [
                ("prep", TARTE_TablePreprocessor()),
                (
                    "tabenc",
                    TARTE_TableEncoder(
                        layer_index=layer_index,
                        device=device,
                    ),
                ),
                (
                    "estimator",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    @property
    def classifier(self) -> XGBClassifier:
        """Return the fitted pipeline's XGBoost classifier.

        input:
            - None: None

        output:
            - classifier: xgboost.XGBClassifier
        """
        return self.pipeline.named_steps["estimator"]

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
    ) -> TARTEFeaturizerEvaluator:
        """Fit the TARTE preprocessing and XGBoost pipeline.

        input:
            - X_train: pandas.DataFrame | numpy.ndarray
            - y_train: numpy.ndarray

        output:
            - evaluator: TARTEFeaturizerEvaluator
        """
        self.pipeline.fit(X_train, y_train)
        return self


    def predict_with_probabilities(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted labels and positive-class probabilities.

        input:
            - X: pandas.DataFrame | numpy.ndarray

        output:
            - predictions_and_probabilities: tuple[numpy.ndarray, numpy.ndarray]
        """
        features = self.pipeline[:-1].transform(X)
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)[:, 1]
        return np.asarray(predictions, dtype=np.int64), np.asarray(probabilities)

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return predicted class labels.

        input:
            - X: pandas.DataFrame | numpy.ndarray

        output:
            - predictions: numpy.ndarray
        """
        return self.pipeline.predict(X)
