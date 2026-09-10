"""XGBoost classification on frozen, YAGO-pretrained TARTE features."""
from xgboost import XGBClassifier

try:
    from .pretrained import extract_features
except ImportError:
    from pretrained import extract_features


class TARTEFeaturizerEvaluator:
    def __init__(self, backbone, device):
        self.backbone = backbone.eval().requires_grad_(False)
        self.device = device
        self.classifier = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
            eval_metric="logloss", tree_method="hist", random_state=42, n_jobs=1,
        )

    def fit(self, rows, y_train):
        features = extract_features(self.backbone, rows, self.device)
        self.classifier.fit(features, y_train)
        return self

    def predict_proba(self, rows):
        features = extract_features(self.backbone, rows, self.device)
        return self.classifier.predict_proba(features)[:, 1]

    def predict(self, rows):
        return (self.predict_proba(rows) >= 0.5).astype(int)
