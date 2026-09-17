"""
pinn_infer.py
-------------
Inference-only code for GB's trained models (PINN, Random Forest, XGBoost,
MLP, Linear Regression), loaded from bundles/my_model_v1/deploy_bundle.pkl
(joblib-compressed -- the fitted Random Forest alone is ~40MB uncompressed,
~9MB compressed). All five were fit on the SAME standardized feature matrix
(bundle.X_train in dor_ml_pipeline_v2_with_pinn), so they share one
age-injection mechanism: build the row once through the fitted preprocessor,
then overwrite the two standardized age_days/log_age_days columns per
requested age -- no need to re-run the ColumnTransformer per age.

Only main_net is reproduced for the PINN (see module docstring in the
original pinn_model.py) -- the composition sub-network and ODE residual are
training-time-only components, not needed to produce a prediction.

Requires: torch, scikit-learn, xgboost, joblib (see pyproject.toml's
`my_model` extra: `pip install -e .[my_model]`).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MainNet(nn.Module):
    def __init__(self, n_in: int, hidden=(128, 64, 32)):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_full):
        raw = self.net(x_full).squeeze(-1)
        return torch.sigmoid(raw) * 105.0


class MultiModelPredictor:
    """Loads deploy_bundle.pkl once and predicts a DoR(t) curve from any of
    GB's fitted models (pinn / rf / xgb / mlp / linreg) for a single mix row
    built from DoRGems's own feature dict."""

    SKLEARN_MODEL_KEYS = ("linreg", "mlp", "rf", "xgb")

    def __init__(self, bundle_path: Path):
        b = joblib.load(bundle_path)
        self.preprocessor = b["preprocessor"]
        self.numeric_feats: list[str] = b["numeric_feats"]
        self.categorical_feats: list[str] = b["categorical_feats"]

        sd = b["pinn_state_dict"]
        self.n_full = b["n_full"]
        self.pinn_net = MainNet(self.n_full)
        self.pinn_net.load_state_dict(sd["main_net"])
        self.pinn_net.eval()
        self.age_idx = sd["age_idx"]
        self.logage_idx = sd["logage_idx"]
        self.age_mean = sd["age_mean"]
        self.age_scale = sd["age_scale"]
        self.logage_mean = sd["logage_mean"]
        self.logage_scale = sd["logage_scale"]
        self.log_base = sd["log_base"]  # "e" or "10"

        self.sklearn_models = {}
        for key in self.SKLEARN_MODEL_KEYS:
            if key in b:
                self.sklearn_models[key] = b[key]

    def _log_age(self, age_days: np.ndarray) -> np.ndarray:
        age_days = np.clip(age_days, 1e-6, None)
        return np.log10(age_days) if self.log_base == "10" else np.log(age_days)

    def _build_X(self, row: dict, ages) -> np.ndarray:
        """Row -> preprocessor.transform once, then one standardized feature
        matrix with age_days/log_age_days overwritten per requested age."""
        ages = np.asarray(ages, dtype=float)
        cols = self.numeric_feats + self.categorical_feats
        row = dict(row)
        row.setdefault("age_days", 28.0)
        row.setdefault("log_age_days", self._log_age(np.array([28.0]))[0])
        df = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])

        X = self.preprocessor.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        X = np.repeat(X, len(ages), axis=0)

        log_ages = self._log_age(ages)
        X[:, self.age_idx] = (ages - self.age_mean) / self.age_scale
        X[:, self.logage_idx] = (log_ages - self.logage_mean) / self.logage_scale
        return X

    def predict_curve(self, row: dict, ages, model: str = "pinn") -> np.ndarray:
        """model: 'pinn', 'rf', 'xgb', 'mlp', or 'linreg'."""
        X = self._build_X(row, ages)
        if model == "pinn":
            with torch.no_grad():
                out = self.pinn_net(torch.tensor(X, dtype=torch.float32)).numpy()
            return out
        if model in self.sklearn_models:
            return np.asarray(self.sklearn_models[model].predict(X), dtype=float)
        raise ValueError(f"unknown or unavailable model: {model!r} (have: pinn, {list(self.sklearn_models)})")


# Backwards-compatible alias (earlier version of this module only exposed
# the PINN predictor under this name).
PinnPredictor = MultiModelPredictor
