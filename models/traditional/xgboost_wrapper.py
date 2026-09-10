"""Cross-validated XGBoost wrapper for the traditional-model benchmark."""

from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


RANDOM_STATE = 42
CV_FOLDS = 5

# GridSearchCV evaluates every configuration with the complete benchmark metric
# set, while macro-F1 determines which fitted model is retained.
SCORING = {
    "f1_macro": make_scorer(
        f1_score, average="macro", zero_division=0
    ),
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "auc": "roc_auc",
}


class XGBoostEvaluator:
    """Tune XGBoost on training folds and expose the selected estimator."""

    model_name = "xgboost"

    def __init__(
        self,
        random_state=RANDOM_STATE,
        cv_folds=CV_FOLDS,
        search_jobs=-1,
    ):
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.search_jobs = search_jobs
        self.search_ = None

    def _build_search(self):
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
            scoring=SCORING,
            refit="f1_macro",
            cv=cross_validation,
            n_jobs=self.search_jobs,
        )

    def fit(self, X_train, y_train):
        self.search_ = self._build_search()
        self.search_.fit(X_train, y_train)
        return self

    def _require_fitted(self):
        if self.search_ is None or not hasattr(self.search_, "best_estimator_"):
            raise RuntimeError("Call fit before requesting XGBoost results.")

    @property
    def best_parameters(self):
        self._require_fitted()
        return self.search_.best_params_

    def cross_validation_metrics(self):
        """Return mean validation metrics for the selected configuration."""
        self._require_fitted()
        best_index = self.search_.best_index_
        return {
            metric: self.search_.cv_results_[f"mean_test_{metric}"][best_index]
            for metric in SCORING
        }

    def predict_proba(self, X):
        """Return positive-class probabilities from the selected estimator."""
        self._require_fitted()
        return self.search_.best_estimator_.predict_proba(X)[:, 1]
