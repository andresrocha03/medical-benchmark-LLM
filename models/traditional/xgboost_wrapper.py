"""Cross-validated XGBoost wrapper for the traditional-model benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from models.utils import CROSS_VALIDATION_SCORING


RANDOM_STATE = 42
CV_FOLDS = 5


class XGBoostEvaluator:
    """Tune XGBoost on training folds and expose the selected estimator."""

    model_name = "xgboost"

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
        cv_folds: int = CV_FOLDS,
        search_jobs: int = -1,
    ) -> None:
        """Configure the XGBoost cross-validation search.

        input:
            - random_state: int
            - cv_folds: int
            - search_jobs: int

        output:
            - None: None
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.search_jobs = search_jobs
        self.search_ = None

    def _build_search(self) -> GridSearchCV:
        """Build the configured XGBoost grid search.

        input:
            - None: None

        output:
            - search: sklearn.model_selection.GridSearchCV
        """
        # Keep each XGBoost fit single-threaded because GridSearchCV parallelizes
        # across parameter combinations and validation folds.
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=1,
        )
        parameter_grid = {
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
        }
        cross_validation = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        return GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring=CROSS_VALIDATION_SCORING,
            refit="f1_macro",
            cv=cross_validation,
            n_jobs=self.search_jobs,
        )

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
    ) -> XGBoostEvaluator:
        """Fit the XGBoost grid search on training data.

        input:
            - X_train: pandas.DataFrame | numpy.ndarray
            - y_train: numpy.ndarray

        output:
            - evaluator: XGBoostEvaluator
        """
        self.search_ = self._build_search()
        self.search_.fit(X_train, y_train)
        return self

    def _require_fitted(self) -> None:
        """Raise an error when the evaluator has not been fitted.

        input:
            - None: None

        output:
            - None: None
        """
        if self.search_ is None or not hasattr(self.search_, "best_estimator_"):
            raise RuntimeError("Call fit before requesting XGBoost results.")

    @property
    def best_parameters(self) -> dict[str, Any]:
        """Return the selected XGBoost hyperparameters.

        input:
            - None: None

        output:
            - best_parameters: dict[str, Any]
        """
        self._require_fitted()
        return self.search_.best_params_

    def cross_validation_metrics(self) -> dict[str, float]:
        """Return validation metrics for the selected configuration.

        input:
            - None: None

        output:
            - validation_metrics: dict[str, float]
        """
        self._require_fitted()
        best_index = self.search_.best_index_
        return {
            metric: self.search_.cv_results_[f"mean_test_{metric}"][best_index]
            for metric in CROSS_VALIDATION_SCORING
        }

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return positive-class probabilities from the selected estimator.

        input:
            - X: pandas.DataFrame | numpy.ndarray

        output:
            - positive_probabilities: numpy.ndarray
        """
        self._require_fitted()
        return self.search_.best_estimator_.predict_proba(X)[:, 1]
