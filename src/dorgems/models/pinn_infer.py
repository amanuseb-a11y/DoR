"""
pinn_infer.py
-------------
Inference-only port of the MainNet half of GB's PINN (from
dor_ml_pipeline_v2_with_pinn/pinn_model.py). Only main_net is needed to
predict a DoR(t) curve -- the composition sub-network and the ODE residual
are training-time-only components (used to shape the loss, not to produce a
prediction), so they are intentionally NOT reproduced here.

Architecture must match pinn_model.py:MainNet exactly, or loading the
state_dict will fail.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
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


class PinnPredictor:
    """Loads deploy_bundle.pkl (preprocessor + pinn_state_dict + feature
    lists) and predicts DoR(%) at arbitrary ages for a single mix row built
    from DoRGems's own feature dict."""

    def __init__(self, bundle_path: Path):
        with open(bundle_path, "rb") as f:
            b = pickle.load(f)
        self.preprocessor = b["preprocessor"]
        self.numeric_feats: list[str] = b["numeric_feats"]
        self.categorical_feats: list[str] = b["categorical_feats"]
        sd = b["pinn_state_dict"]
        self.n_full = b["n_full"]
        self.main_net = MainNet(self.n_full)
        self.main_net.load_state_dict(sd["main_net"])
        self.main_net.eval()
        self.age_idx = sd["age_idx"]
        self.logage_idx = sd["logage_idx"]
        self.age_mean = sd["age_mean"]
        self.age_scale = sd["age_scale"]
        self.logage_mean = sd["logage_mean"]
        self.logage_scale = sd["logage_scale"]
        self.log_base = sd["log_base"]  # "e" or "10"

    def _log_age(self, age_days: np.ndarray) -> np.ndarray:
        age_days = np.clip(age_days, 1e-6, None)
        return np.log10(age_days) if self.log_base == "10" else np.log(age_days)

    def predict_curve(self, row: dict, ages) -> np.ndarray:
        """row: dict with keys covering (a subset of) numeric_feats +
        categorical_feats; missing keys are left NaN and imputed by the
        fitted preprocessor exactly as at training time. ages: iterable of
        age_days values (any placeholder value works for age_days/
        log_age_days in `row` -- both are overwritten below per requested
        age)."""
        import pandas as pd

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

        with torch.no_grad():
            out = self.main_net(torch.tensor(X, dtype=torch.float32)).numpy()
        return out
