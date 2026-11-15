"""Two separate predictions per stock, each calibrated and confidence-checked on history:
  1d  : will the stock close higher tomorrow?                    (absolute direction)
  10d : will it beat the median TA-125 stock over 10 trading days? (relative to the market)
A prediction is only FLAGGED high-confidence if its calibrated probability reaches the target AND a hold-out of past out-of-sample predictions
confirmed that level. Calibration is refit about monthly and cached in data/calibration/."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as C
from .forward import oos_predictions
from .model import predict, train

CAL_DIR = Path("data/calibration")
# Each question is answered by an ENSEMBLE: the average probability of three LightGBM / HistGB models that look at the stocks differently
# (base features; + overnight macro and extra analysis features; + the Alpha158 family). Chosen from the experiments in README (validation 2023-24,
# one look at the untouched 2025-26 test): the full ensembles, not the best-looking subset.
SPECS = {
    "1d": dict(target="label_up1", date_col="label_date1", ret_col="ret_fwd1", min_hit=0.55, title="Up tomorrow",
               members=[dict(algo="lgbm", feature_set="xs"), dict(algo="lgbm", feature_set="xsx"), dict(algo="lgbm", feature_set="xsxa")]),
    "10d": dict(target="label_xs", date_col="label_date", ret_col="excess_ret", min_hit=0.55, title="Beats the market over 10 days",
                members=[dict(algo="hgb", feature_set="xsx"), dict(algo="lgbm", feature_set="xsx"), dict(algo="lgbm", feature_set="xsxa")]),
}


def signature(key: str) -> str:
    """Short id of the recipe: a cached calibration is only valid for the models that produced it."""
    import hashlib
    s = SPECS[key]
    return hashlib.md5(repr((s["target"], s["members"])).encode()).hexdigest()[:8]


def ensemble_oos(ds: pd.DataFrame, start: str, end: str, key: str, retrain_every: int = 63, window_years: int | None = 6) -> pd.DataFrame:
    """Walk-forward out-of-sample predictions of the ensemble: the members' probabilities averaged per (date, ticker). Columns: date, ticker, p, y, ret."""
    s = SPECS[key]; parts = []
    for m in s["members"]:
        o = oos_predictions(ds, start, end, s["target"], s["date_col"], s["ret_col"], retrain_every=retrain_every, window_years=window_years, **m)
        parts.append(o.set_index(["date", "ticker"]))
    out = parts[0][["y", "ret"]].copy()
    out["p"] = pd.concat([p["p"] for p in parts], axis=1, join="inner").mean(axis=1)
    return out.dropna().reset_index()


def get_calibration(ds: pd.DataFrame, key: str, force: bool = False, years: int = 2, min_n: int = 300,
                    cal_dir: Path = CAL_DIR, retrain_every: int = 63) -> C.Calibration:
    path = cal_dir / f"{key}_{signature(key)}.json"
    if not force:
        cal = C.load(path)
        if cal is not None:
            return cal
    s = SPECS[key]
    end = ds.loc[ds[s["target"]].notna(), "date"].max()
    start = end - pd.DateOffset(years=years)
    oos = ensemble_oos(ds, str(start.date()), str(end.date()), key, retrain_every)
    cal = C.fit_calibration(oos, s["min_hit"], min_n)
    C.save(cal, path)
    return cal


def dual_predictions(ds: pd.DataFrame, asof: str | None = None, window_years: int = 6, cal_dir: Path = CAL_DIR,
                     force_calibration: bool = False) -> tuple[pd.DataFrame, dict]:
    """Returns (frame with p1_raw, p1, flag1, p10_raw, p10, flag10 per ticker, {key: Calibration}). Raw = the ensemble's average probability."""
    d = pd.Timestamp(asof) if asof else ds["date"].max()
    rows = ds[ds["date"] == d].dropna(subset=["rsi14", "dist_ma50"]).copy()
    out = pd.DataFrame({"ticker": rows["ticker"].to_numpy()})
    cals = {}
    for key, s in SPECS.items():
        raws = []
        for m in s["members"]:
            model = train(ds, d, window_years, target=s["target"], date_col=s["date_col"], **m)
            raws.append(predict(model, rows).to_numpy())
        raw = np.mean(raws, axis=0)
        cal = get_calibration(ds, key, force=force_calibration, cal_dir=cal_dir)
        p = cal.apply(raw)
        suffix = "1" if key == "1d" else "10"
        out[f"p{suffix}_raw"], out[f"p{suffix}"], out[f"flag{suffix}"] = raw, p, cal.flag(p)
        cals[key] = cal
    return out, cals


def merge_into_scan(scan: pd.DataFrame, dual: pd.DataFrame) -> pd.DataFrame:
    return scan.merge(dual, on="ticker", how="left")


def calibration_summary(cals: dict, L=None) -> str:
    """Plain-language facts about what the probabilities can honestly claim (English or Hebrew)."""
    from .i18n import Lang
    L = L or Lang("en")
    lines = [L.t("Calibration (probabilities checked against what actually happened on a held-out stretch of history):")]
    for key, cal in cals.items():
        s = cal.stats; title = L.t(SPECS[key]["title"])
        if cal.t_up is None and cal.t_down is None:
            flagged = L.t("no prediction reached the confidence bar in the hold-out, so none are flagged")
        else:
            flagged = (L.t("flag 'up' at >= {t} calibrated", t=f"{cal.t_up:.0%}") if cal.t_up else L.t("no 'up' flag")) + \
                      (L.t(", 'down' at <= {t}", t=f"{cal.t_down:.0%}") if cal.t_down else L.t(", no 'down' flag"))
        lines.append(L.t("- {title}: highest calibrated probability the model ever gave was {mx} (lowest {mn}); calibration error {er} raw -> {ec} calibrated; base rate {base}; {flagged}.",
                         title=title, mx=f"{s['max_calibrated_p']:.0%}", mn=f"{s['min_calibrated_p']:.0%}", er=f"{s['ece_raw']:.3f}",
                         ec=f"{s['ece_calibrated']:.3f}", base=f"{s['base_rate_holdout']:.1%}", flagged=flagged))
    return "\n".join(lines)
