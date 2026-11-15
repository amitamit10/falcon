"""Generic walk-forward out-of-sample predictions and honest evaluation for any target (next-day up, 10-day beat-median, ...).
Every prediction is made by a model trained ONLY on rows whose label was already known on the prediction date."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .model import predict, train


def oos_predictions(ds: pd.DataFrame, start: str, end: str, target: str = "label_up1", date_col: str = "label_date1",
                    ret_col: str = "ret_fwd1", retrain_every: int = 21, window_years: int | None = 6, **train_kw) -> pd.DataFrame:
    """Columns: date, ticker, p (raw model score), y (outcome 0/1), ret (forward return)."""
    dates = sorted(ds.loc[(ds["date"] >= start) & (ds["date"] <= end), "date"].unique())
    parts = []
    for i in range(0, len(dates), retrain_every):
        block = dates[i:i + retrain_every]
        try:
            m = train(ds, pd.Timestamp(block[0]), window_years, target=target, date_col=date_col, **train_kw)
        except ValueError:
            continue
        rows = ds[ds["date"].isin(block) & ds[target].notna()].dropna(subset=["rsi14", "dist_ma50"]).copy()
        rows["p"] = predict(m, rows)
        parts.append(rows[["date", "ticker", "p", target, ret_col]].rename(columns={target: "y", ret_col: "ret"}))
    if not parts:
        raise ValueError("no out-of-sample predictions produced; check the date range")
    return pd.concat(parts, ignore_index=True)


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted probability and actual hit rate, weighted by bin size."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def evaluate(oos: pd.DataFrame, horizon: int = 1, min_stocks: int = 30) -> dict:
    """Ranking and probability quality of out-of-sample predictions. Rank IC uses non-overlapping dates."""
    d = oos.dropna(subset=["p", "y"])
    ic = []
    for _, g in d[d["date"].isin(sorted(d["date"].unique())[::max(1, horizon)])].groupby("date"):
        if len(g) >= min_stocks:
            ic.append(g["p"].rank().corr(g["ret"].rank()))
    ic = pd.Series(ic)
    q = pd.qcut(d["p"].rank(method="first"), 10, labels=False)
    top, bot = d[q == 9]["y"].mean(), d[q == 0]["y"].mean()
    return {"n": len(d), "base_rate": round(float(d["y"].mean()), 4), "auc": round(float(roc_auc_score(d["y"], d["p"])), 4),
            "brier": round(brier(d["p"], d["y"]), 5), "ece": round(ece(d["p"], d["y"]), 4),
            "rank_IC": round(float(ic.mean()), 4), "IC_t": round(_t(ic), 2),
            "top_decile_hit": round(float(top), 4), "bottom_decile_hit": round(float(bot), 4)}
