"""GB's trained models, wired into DoRGems (Step 12+).

Loads bundles/my_model_v1/deploy_bundle.pkl -- five of GB's own 7-model
benchmarking pipeline (dor_ml_pipeline_v2_with_pinn), fit on
ml_master_DoR_dataset_v13_1.xlsx: PINN (physics-informed, best performer),
XGBoost, Random Forest, MLP, and Linear Regression (the simplest baseline).
Confirmed Test R^2 on the full 7-model benchmark: linreg 0.616, rf 0.783,
xgb 0.835, mlp 0.832, cnn 0.746, grnn 0.816, pinn 0.844 -- only CNN and GRNN
are not wired in live here (each has its own bespoke architecture that would
need its own inference-only port, same treatment PINN's main_net got).

This module is entirely self-contained inside DoR/ (no dependency on the
Desktop pipeline folder at runtime) -- everything it needs (the fitted
preprocessor, the trained network weights, the feature-name lists) was
packaged once into bundles/my_model_v1/deploy_bundle.pkl.

INPUT: this consumes the exact same `feats` dict that db.features.
scm_input_to_features() produces (already computed once in predict.py),
PLUS (new) the MixSpec itself, so it can read `mix.opc_oxides` -- the
cement's own oxide composition, which DoRGems's own feats dict does NOT
carry (scm_input_to_features() never reads it), but which pilot/schemas.py's
MixSpec supports and templates/example inputs actually populate.

KNOWN LIMITATION (flagged, not hidden): when `mix.opc_oxides` is not given,
the pipeline's binder_oxide_*/system_* columns (which need the cement's own
chemistry) fall back to NaN -> the SAME median imputer used at training
time, not a guess coded here. LOI and TiO2 specifically are commonly absent
even when opc_oxides IS given (templates/mix.yaml's example only lists 8 of
the 10 oxides) and stay NaN/imputed regardless. The literature
measurement-method columns (dor_method_group/confidence/is_direct) have no
meaning for a brand-new prediction and are always left NaN/imputed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import repo_root

_PREDICTOR = None

# DoRGems role (db.features.role_from_text) -> GB pipeline's scm_type
# category (as actually fit: ['calcined_clay','fly_ash','metakaolin',
# 'other_scms','silica_fume','slag'] -- natural_pozzolan/limestone/
# rice_husk_ash/glass_powder have no dedicated class in this fitted model,
# so they fold into 'other_scms').
SCM_ROLE_TO_TYPE = {
    "fly_ash": "fly_ash",
    "slag": "slag",
    "silica_fume": "silica_fume",
    "metakaolin": "metakaolin",
    "calcined_clay": "calcined_clay",
    "natural_pozzolan": "other_scms",
    "limestone": "other_scms",
    "rice_husk_ash": "other_scms",
    "glass_powder": "other_scms",
}

_OXIDE_KEYS = ["CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "SO3", "Na2O", "K2O", "TiO2", "LOI"]
_DOSE_KEYS = ["CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "SO3"]
_SYSTEM_KEYS = ["CaO", "SiO2", "Al2O3", "Fe2O3", "MgO", "SO3", "Na2O", "K2O", "TiO2", "LOI"]


MODEL_TEST_R2 = {
    "pinn": 0.844,
    "xgb": 0.835,
    "mlp": 0.832,
    "rf": 0.783,
    "grnn": 0.816,
    "cnn": 0.746,
    "linreg": 0.616,
}
MODEL_LABELS = {
    "pinn": "PINN (physics-informed)",
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "mlp": "MLP",
    "linreg": "Linear Regression",
}
DEFAULT_MODELS = ("pinn", "xgb", "rf", "mlp", "linreg")


def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from .pinn_infer import MultiModelPredictor

        bundle_path = repo_root() / "bundles" / "my_model_v1" / "deploy_bundle.pkl"
        if not bundle_path.is_file():
            raise FileNotFoundError(f"my_model bundle not found at {bundle_path}")
        _PREDICTOR = MultiModelPredictor(bundle_path)
    return _PREDICTOR


def _num(feats: dict, key: str) -> float | None:
    v = feats.get(key)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return float(v)


def _get_opc_oxides(mix: Any) -> dict[str, float]:
    """Pull opc_oxides off a MixSpec (pydantic model or plain dict). Returns
    {} if not given -- callers then leave binder_oxide_*/system_* as NaN,
    exactly as before this was wired in."""
    if mix is None:
        return {}
    if isinstance(mix, dict):
        ox = mix.get("opc_oxides")
    else:
        ox = getattr(mix, "opc_oxides", None)
    return dict(ox) if ox else {}


def feats_to_my_model_row(feats: dict, mix: Any = None) -> dict[str, Any]:
    """Translate a DoRGems scm_input_to_features() dict (+ optionally the
    MixSpec, for opc_oxides) into GB pipeline's raw column schema
    (config.NUMERIC_FEATS / CATEGORICAL_FEATS in
    dor_ml_pipeline_v2_with_pinn/config.py). See module docstring for which
    columns can still be NaN (imputed) and why."""
    scm_pct = _num(feats, "scm_pct")
    frac = (scm_pct / 100.0) if scm_pct is not None else None
    oxide = {k: _num(feats, f"scm_{k}") for k in _OXIDE_KEYS}
    opc = _get_opc_oxides(mix)

    row: dict[str, Any] = {
        "scm_replacement_pct": scm_pct,
        "scm_replacement_fraction": frac,
        "water_binder_ratio": _num(feats, "w_b"),
        "curing_temperature_C": _num(feats, "curing_temp_C"),
        # DoRGems already computes these two ratios from the SCM's own
        # oxides (db/features.py:derive_composition_features) -- identical
        # definition to the pipeline's scm_ratio_* columns, reused directly.
        "scm_ratio_CaO_SiO2": _num(feats, "CaO_SiO2"),
        "scm_ratio_Al2O3_SiO2": _num(feats, "Al_Si"),
        "scm_phys_blaine_m2_per_kg": _num(feats, "scm_blaine_m2_kg"),
        "scm_phys_d50_um": _num(feats, "scm_d50_um"),
        "scm_phys_BET_m2_per_g": _num(feats, "scm_bet_m2_g"),
    }
    for k, v in oxide.items():
        row[f"scm_oxide_{k}_pct"] = v
    for k in _DOSE_KEYS:
        v = oxide.get(k)
        row[f"scm_dose_{k}_pct"] = (v * frac) if (v is not None and frac is not None) else None

    # binder_oxide_* (the OPC's own chemistry) + system_* (mass-weighted
    # blend of OPC and SCM, the standard blended-cement-chemistry
    # construction: system = binder*(1-replacement_fraction) + scm*fraction)
    # -- only computable where opc_oxides supplies that oxide AND both the
    # SCM oxide and replacement fraction are known; otherwise NaN -> imputed,
    # same as before.
    system = {}
    for k in _SYSTEM_KEYS:
        b = opc.get(k)
        row[f"binder_oxide_{k}_pct"] = float(b) if b is not None else None
        s = oxide.get(k)
        if b is not None and s is not None and frac is not None:
            system[k] = float(b) * (1.0 - frac) + s * frac
        else:
            system[k] = None
        row[f"system_{k}_pct"] = system[k]

    def _ratio(a, b):
        return (a / b) if (a is not None and b not in (None, 0)) else None

    row["system_CaO_SiO2"] = _ratio(system.get("CaO"), system.get("SiO2"))
    row["system_Al2O3_SiO2"] = _ratio(system.get("Al2O3"), system.get("SiO2"))
    row["system_CaO_Al2O3"] = _ratio(system.get("CaO"), system.get("Al2O3"))

    row["scm_type"] = SCM_ROLE_TO_TYPE.get(feats.get("scm_role"))
    # binder_type, dor_method_group, dor_method_confidence, dor_method_is_direct:
    # deliberately absent -> NaN -> imputed (no DoRGems equivalent; see docstring).
    return row


def predict_my_model(feats: dict, ages, mix: Any = None) -> np.ndarray:
    """Back-compat: DoR(%) curve from the PINN alone. See predict_my_models()
    for the multi-model version used by predict.py now."""
    predictor = _get_predictor()
    row = feats_to_my_model_row(feats, mix=mix)
    curve = predictor.predict_curve(row, ages, model="pinn")
    return np.clip(curve, 0.0, 100.0)


def predict_my_models(feats: dict, ages, mix: Any = None, models=DEFAULT_MODELS) -> dict[str, Any]:
    """DoR(%) curves from several of GB's trained models at once (same
    row/features, same fitted preprocessor -- see pinn_infer.MultiModelPredictor).
    Returns {model_key: {"alpha_pct": [...], "label": ..., "test_r2": ...}}.
    `models` may include 'pinn', 'linreg', 'mlp' (whatever's in the bundle)."""
    predictor = _get_predictor()
    row = feats_to_my_model_row(feats, mix=mix)
    out: dict[str, Any] = {}
    for key in models:
        curve = np.clip(predictor.predict_curve(row, ages, model=key), 0.0, 100.0)
        out[key] = {
            "alpha_pct": curve.tolist(),
            "label": MODEL_LABELS.get(key, key),
            "test_r2": MODEL_TEST_R2.get(key),
        }
    return out
