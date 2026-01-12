import json
from typing import Dict
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


RANDOM_STATE = 42


class TaxiFareModel:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = RANDOM_STATE,
    ):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.metrics: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            raise ValueError("Input data cannot be empty")
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        predictions = self.predict(X)
        self.metrics = {
            "r2_score": r2_score(y, predictions),
            "rmse": root_mean_squared_error(y, predictions),
            "mae": mean_absolute_error(y, predictions),
        }
        return self.metrics

    def save(self, model_path: str, metrics_path: str) -> None:
        joblib.dump(self.model, model_path)

        metrics_info = {
            "metrics": self.metrics,
            "model_params": self.model.get_params(),
            "n_features": self.model.n_features_in_,
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics_info, f, indent=2)

    @classmethod
    def load(cls, model_path: str) -> "TaxiFareModel":
        instance = cls()
        instance.model = joblib.load(model_path)
        return instance
