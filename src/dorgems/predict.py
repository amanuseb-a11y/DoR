"""Scenario A step 1: SCMSpec + MixSpec + ages → prediction.json (spec §5, §5.6).

Deterministic: same input, same bundle, same seed → same output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import PREDICTION_SCHEMA, __version__
from .config import configs_dir, literature_db_path
from .db.features import scm_input_to_features
from .models import ood as ood_mod
from .models.bayes import predict_curve
from .models.bundle import Bundle, load_bundle
from .models.ensemble import DEFAULT_ANCHORS, ENSEMBLE_MODES, fit_stretched_exp, reweight
from .models.gbm import predict_points


def load_defaults() -> dict[str, Any]:
    p = configs_dir() / "defaults.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _plain(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _hash_input(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def predict(
    scm: Any,
    mix: Any,
    ages: list[float] | None = None,
    *,
    bundle: Bundle | None = None,
    ensemble: str | None = None,
    method_group: str | None = None,
    seed: int = 0,
    db_path: str | Path | None = None,
    n_analogues: int = 5,
    prediction_id: str | None = None,
) -> dict[str, Any]:
    from .pilot.schemas import coerce_mix, coerce_scm

    scm_m, mix_m = coerce_scm(scm), coerce_mix(mix)
    defaults = load_defaults()
    ens_cfg = defaults.get("ensemble", {})
    mode = ensemble or ens_cfg.get("mode", "blend")
    if mode not in ENSEMBLE_MODES:
        raise ValueError(f"ensemble must be one of {ENSEMBLE_MODES}")
    ages_arr = np.asarray(ages if ages is not None else defaults.get("ages_d_default", [1, 3, 7, 28, 90, 180, 365]), float)
    if np.any(ages_arr <= 0):
        raise ValueError("ages must be positive")
    bundle = bundle or load_bundle(require_gbm=(mode != "bayes"))
    warnings: list[str] = []

    feats = scm_input_to_features(scm_m, mix_m)
    x_bayes = {k: feats.get(k) for k in bundle.bayes.feats}
    role = scm_m.role
    if scm_m.amorphous_pct is not None:
        warnings.append("amorphous_pct given: must be a measured value, not 100 − Σ crystalline (spec §14)")

    # --- Bayes ---------------------------------------------------------------
    draws = predict_curve(bundle.bayes, x_bayes, role, ages_arr, new_study=True, method_group=method_group, rng_seed=seed)
    bayes_summary = draws.summary(rng_seed=seed)
    bayes_block = {**bayes_summary, "role_used": draws.role_used, "flags": list(draws.flags), "imputed": list(draws.imputed), "n_draws": int(draws.a_max.shape[0])}
    warnings += [f"bayes:{f}" for f in draws.flags]

    # --- GBM ------------------------------------------------------------------
    gbm_block: dict[str, Any] | None = None
    anchors = np.asarray(ens_cfg.get("anchors_d", DEFAULT_ANCHORS), float)
    if bundle.gbm is not None:
        p_ages, ginfo = predict_points(bundle.gbm, feats, role, ages_arr, method_group=method_group or "missing")
        p_anch, _ = predict_points(bundle.gbm, feats, role, anchors, method_group=method_group or "missing")
        sigma_g = float(bundle.gbm.sigma_point_pct or ens_cfg.get("sigma_point_pct", 12.0))
        gbm_block = {"alpha_pct": p_ages.tolist(), "anchors_d": anchors.tolist(), "anchors_pct": p_anch.tolist(), "method_group_used": ginfo["method_group_used"], "role_used": ginfo["role_used"], "sigma_point_pct": sigma_g, "flags": ginfo["flags"]}
        warnings += [f"gbm:{f}" for f in ginfo["flags"]]
    elif mode != "bayes":
        warnings.append("gbm bundle missing; ensemble falls back to bayes")
        mode = "bayes"

    # --- GB's trained model (PINN; additional 8th source, not yet blended
    # into 'ensemble'/'recommended' -- see models/my_model.py docstring for
    # what it is and its known input-schema limitations). Never fatal: if the
    # bundle or torch isn't available, predict() still returns the original
    # bayes/gbm/ensemble result untouched, just with a warning added. -------
    my_model_block: dict[str, Any] | None = None
    try:
        from .models.my_model import predict_my_models

        my_model_block = {
            "models": predict_my_models(feats, ages_arr, mix=mix_m),
            "source": "GB, ml_master_DoR_dataset_v13_1.xlsx",
            "note": "Not yet blended into 'recommended' or 'ensemble' -- reported alongside for comparison.",
        }
    except Exception as e:  # noqa: BLE001
        warnings.append(f"my_model unavailable: {e}")

    # --- ensemble -------------------------------------------------------------
    ens_block: dict[str, Any] = {"mode": mode}
    recommended_source = "bayes"
    if mode == "blend" and gbm_block is not None:
        new, info = reweight(draws, anchors, np.asarray(gbm_block["anchors_pct"]), sigma_g=gbm_block["sigma_point_pct"], beta=bundle.bayes.beta_shape, ess_min=float(ens_cfg.get("ess_min", 100)), rng_seed=seed)
        ens_block.update({"ess": info["ess"], "applied": info["applied"], "sigma_g": info["sigma_g"]})
        warnings += info["warnings"]
        if info["applied"]:
            ens_block.update(new.summary(rng_seed=seed))
            recommended_source = "ensemble"
        else:
            ens_block.update(bayes_summary)
            ens_block["mode"] = "bayes(fallback)"
    elif mode == "gbm_anchor_only" and gbm_block is not None:
        fit = fit_stretched_exp(anchors, np.asarray(gbm_block["anchors_pct"]), beta=bundle.bayes.beta_shape)
        curve = fit["a_max"] * (1 - np.exp(-((ages_arr / fit["tau_d"]) ** fit["beta"])))
        ens_block.update({"a_max": {"q50": fit["a_max"]}, "tau_d": {"q50": fit["tau_d"]}, "alpha_pct": {"latent": {"q50": np.clip(curve, 0, 100).tolist()}}, "fit_rmse_pct": fit["rmse_pct"]})
        recommended_source = "ensemble"
        warnings.append("gbm_anchor_only: no uncertainty (diagnostic mode)")
    else:
        ens_block.update(bayes_summary)

    src_block = ens_block if recommended_source == "ensemble" else bayes_block
    lat = src_block["alpha_pct"]["latent"]
    recommended = {
        "source": recommended_source,
        "alpha_pct_q50": lat["q50"],
        "alpha_pct_q05": lat.get("q05", lat["q50"]),
        "alpha_pct_q95": lat.get("q95", lat["q50"]),
        "a_max_pct_q50": src_block["a_max"]["q50"],
        "tau_d_q50": src_block["tau_d"]["q50"],
    }

    # --- OOD + evidence ---------------------------------------------------------
    ood = ood_mod.assess(bundle.ood, role, feats)
    evidence: dict[str, Any] = {"analogues": [], "n_mixes": 0, "n_papers": 0, "flags": []}
    dbp = Path(db_path) if db_path else literature_db_path(required=False)
    if dbp and Path(dbp).is_file():
        from .db.analogues import find_analogues
        from .db.reader import open_ro

        con = open_ro(dbp)
        try:
            ev = find_analogues(con, feats, role, k=n_analogues, cache_key=str(dbp))
            evidence = {"analogues": ev["mixes"], "n_mixes": ev["n_mixes"], "n_papers": ev["n_papers"], "flags": ev["flags"], "db": str(dbp)}
        finally:
            con.close()
    else:
        warnings.append("literature DB not available: no analogue evidence attached")

    input_payload = {"scm": _plain(scm_m), "mix": _plain(mix_m), "ages_d": ages_arr.tolist(), "ensemble": mode, "method_group": method_group, "seed": seed}
    pid = prediction_id or _hash_input(input_payload)
    imputed = list(draws.imputed)
    features_block = {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)) for k, v in feats.items() if k not in ("scm_role", "method_group")}
    features_block["imputed"] = imputed

    return {
        "schema": PREDICTION_SCHEMA,
        "id": pid,
        "input": input_payload,
        "features": features_block,
        "role_bayes": draws.role_used,
        "role_gbm": gbm_block["role_used"] if gbm_block else None,
        "beta_shape": bundle.bayes.beta_shape,
        "bayes": bayes_block,
        "gbm": gbm_block,
        "my_model": my_model_block,
        "ensemble": ens_block,
        "recommended": recommended,
        "ood": ood,
        "evidence": evidence,
        "warnings": warnings,
        "provenance": {"dorgems": __version__, **bundle.provenance, "seed": seed, "input_hash": _hash_input(input_payload)},
    }


def write_prediction(pred: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "prediction.json"
    p.write_text(json.dumps(pred, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
