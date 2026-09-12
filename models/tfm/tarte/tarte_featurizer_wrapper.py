"""XGBoost classification on features from the official frozen TARTE model."""

import numpy as np
from sklearn.pipeline import Pipeline
from tarte_ai import TARTE_TableEncoder, TARTE_TablePreprocessor
from xgboost import XGBClassifier


class TARTEFeaturizerEvaluator:
    """Official frozen TARTE featurizer followed by XGBoost."""

    def __init__(self, device="cpu", layer_index=2):
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
    def classifier(self):
        return self.pipeline.named_steps["estimator"]

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)
        return self

    def predict_with_probabilities(self, X):
        """Transform once, then return labels and positive-class probabilities."""
        features = self.pipeline[:-1].transform(X)
        predictions = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)[:, 1]
        return np.asarray(predictions, dtype=np.int64), np.asarray(probabilities)

    def predict(self, X):
        return self.pipeline.predict(X)
