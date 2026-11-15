"""Pooled classifier with strict point-in-time training. Default target: beat the median stock over the horizon
(label_xs); chosen on a validation period and confirmed on an untouched test period (see README)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .features import FEATURE_COLUMNS, build_features, rule_score


def build_dataset(universe: dict[str, pd.DataFrame], benchmark: pd.Series | None = None,
                  horizon: int = 1, pit_top: int | None = None) -> pd.DataFrame:
    """Long frame: date, ticker, features, next_ret, label, label_date.

    horizon = trading days ahead (1 = next day, 5-20 = swing). label_date = date at which the
    label becomes known (t+horizon). Used to avoid look-ahead."""
    parts = []
    for tk, df in universe.items():
        if len(df) < 260:
            continue
        f = build_features(df, benchmark)
        nxt = df["close"].shift(-horizon) / df["close"] - 1
        # Data-quality guard: |move| > 100% in the horizon is a bad bar (unit/split glitch), not a real return
        nxt = nxt.where(nxt.abs() <= 1.0)
        f["next_ret"] = nxt
        f["label"] = (nxt > 0).astype(float).where(nxt.notna())
        f["label_date"] = pd.Series(df.index, index=df.index).shift(-horizon)
        # next-day target (absolute: does the stock close higher tomorrow?) for the separate next-day model
        n1 = (df["close"].shift(-1) / df["close"] - 1)
        n1 = n1.where(n1.abs() <= 1.0)
        f["ret_fwd1"] = n1
        f["label_up1"] = (n1 > 0).astype(float).where(n1.notna())
        f["label_date1"] = pd.Series(df.index, index=df.index).shift(-1)
        f["ticker"] = tk
        f["close"] = df["close"]                                           # report columns, not model features
        from .patterns import all_patterns
        _p = all_patterns(df)
        f["trendline"], f["cup_target"] = _p["trendline"], _p["cup_target"]   # report-only levels
        f["turnover20"] = (df["close"] * df["volume"]).rolling(20).median()
        f["turnover60"] = (df["close"] * df["volume"]).rolling(60).median()    # point-in-time liquidity (known at t)
        f["rule_score"] = f.apply(rule_score, axis=1)
        parts.append(f)
    if not parts:
        raise ValueError("no ticker has enough history (need >= 260 daily bars)")
    out = pd.concat(parts)
    out.index.name = "date"
    out = out.reset_index()
    if pit_top:
        # Point-in-time universe: keep, at every date, only the pit_top most liquid stocks by TRAILING 60d turnover.
        # Selecting by past liquidity (not by today's size) removes most of the "survived to be big" look-ahead.
        rank = out.groupby("date")["turnover60"].rank(ascending=False, method="first")
        out = out[(rank <= pit_top) & out["turnover60"].notna()].copy()
    return add_cross_section(out)


XS_RANK = ["ret_5d", "ret_20d", "rsi14", "dist_ma50", "dist_ma150", "vol_ratio", "dist_52w_high", "atr_pct", "macd_hist_pct"]
XS_RANK2 = ["mom_12_1", "ret_60d", "vol_20d", "illiq_20d", "beta_60d", "resid_mom_60d", "dist_60d_high", "turnover_trend", "skew_60d"]
NEW_FEATURES = ["ret_2d", "ret_60d", "ret_120d", "mom_12_1", "vol_20d", "vol_60d", "vol_ratio_20_60", "skew_60d", "dist_60d_high",
                "dist_60d_low", "up_days_20", "illiq_20d", "turnover_trend", "beta_60d", "resid_mom_60d"]
MARKET_COLUMNS = ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_20d", "mkt_breadth_ma50", "mkt_breadth_ma150", "mkt_dispersion"]
XS_COLUMNS = [f"xs_{c}" for c in XS_RANK]
XS2_COLUMNS = [f"xs_{c}" for c in XS_RANK2]


def add_cross_section(ds: pd.DataFrame) -> pd.DataFrame:
    """Market-wide and cross-sectional columns. Everything at date t uses only rows of date t (known at the close),
    except the label columns, which look forward by design and are only used as targets."""
    g = ds.groupby("date")
    ds["mkt_ret_1d"] = g["ret_1d"].transform("mean")
    ds["mkt_ret_5d"] = g["ret_5d"].transform("mean")
    ds["mkt_ret_20d"] = g["ret_20d"].transform("mean")
    ds["mkt_breadth_ma50"] = (ds["dist_ma50"] > 0).astype(float).where(ds["dist_ma50"].notna()).groupby(ds["date"]).transform("mean")
    ds["mkt_breadth_ma150"] = (ds["dist_ma150"] > 0).astype(float).where(ds["dist_ma150"].notna()).groupby(ds["date"]).transform("mean")
    ds["mkt_dispersion"] = g["ret_1d"].transform("std")
    for c in XS_RANK + XS_RANK2:
        ds[f"xs_{c}"] = g[c].rank(pct=True)
    # cross-sectional target: did the stock beat the median stock over the horizon (removes market-wide moves)
    med = ds.groupby("date")["next_ret"].transform("median")
    ds["excess_ret"] = ds["next_ret"] - med
    ds["label_xs"] = (ds["excess_ret"] > 0).astype(float).where(ds["excess_ret"].notna())
    ds["rank_target"] = ds.groupby("date")["next_ret"].rank(pct=True)          # for the rank-regression variant
    ds["rank_q"] = np.floor(ds["rank_target"] * 5).clip(upper=4)               # 0..4 relevance for the ranking objective (needs numpy import)
    return ds


def feature_columns(feature_set: str) -> list[str]:
    """base = stock-level features; xs = base + market-wide + cross-sectional ranks; xs_only = ranks + market only."""
    if feature_set == "base":
        return list(FEATURE_COLUMNS)
    if feature_set == "xs":
        return list(FEATURE_COLUMNS) + MARKET_COLUMNS + XS_COLUMNS
    if feature_set == "xs_only":
        return MARKET_COLUMNS + XS_COLUMNS
    if feature_set == "xsm":           # xs + overnight cross-market features (needs macro.add_macro on the dataset)
        from .macro import MACRO_COLUMNS
        return list(FEATURE_COLUMNS) + MARKET_COLUMNS + XS_COLUMNS + MACRO_COLUMNS
    if feature_set in ("xse", "xsx"):  # xse = xs + extra analysis features; xsx = xs + macro + extra
        from .extra import EXTRA_COLUMNS
        from .macro import MACRO_COLUMNS
        return list(FEATURE_COLUMNS) + MARKET_COLUMNS + XS_COLUMNS + (MACRO_COLUMNS if feature_set == "xsx" else []) + EXTRA_COLUMNS
    if feature_set in ("xsp", "xspa"):  # xsp = xss + professional chart setups; xspa = xssa + the same
        from .pro import PRO_COLUMNS
        return feature_columns("xss" if feature_set == "xsp" else "xssa") + PRO_COLUMNS
    if feature_set in ("xsf", "xsfa"):  # xsf = xss + quarterly-report features; xsfa = xssa + the same
        from .fundhist import FUND_COLUMNS
        return feature_columns("xss" if feature_set == "xsf" else "xssa") + FUND_COLUMNS
    if feature_set in ("xss", "xssa"):  # xss = xsd + US sector / cross-asset flow; xssa = xsda + the same
        from .sectorflow import SECTOR_COLUMNS
        return feature_columns("xsd" if feature_set == "xss" else "xsda") + SECTOR_COLUMNS
    if feature_set in ("xsd", "xsda"):  # xsd = xsx + cross-listing signals; xsda = xsxa + cross-listing signals
        from .crosslist import CROSS_COLUMNS
        return feature_columns("xsx" if feature_set == "xsd" else "xsxa") + CROSS_COLUMNS
    if feature_set in ("xsa", "xsxa"):  # xsa = xs + alpha158; xsxa = xs + macro + extra + alpha158
        from .alpha158 import ALPHA_COLUMNS
        from .extra import EXTRA_COLUMNS
        from .macro import MACRO_COLUMNS
        base = list(FEATURE_COLUMNS) + MARKET_COLUMNS + XS_COLUMNS + ALPHA_COLUMNS
        return base if feature_set == "xsa" else base + MACRO_COLUMNS + EXTRA_COLUMNS
    if feature_set == "xs2":           # stage 2: + momentum/risk features and their cross-sectional ranks
        return list(FEATURE_COLUMNS) + MARKET_COLUMNS + XS_COLUMNS + XS2_COLUMNS
    raise ValueError(f"unknown feature_set {feature_set}")


DEFAULT_PARAMS = dict(max_depth=4, learning_rate=0.05, max_iter=150, l2_regularization=1.0)
ALGOS = ("hgb", "lgbm", "xgb", "cat", "xgb_big", "cat_big", "ens3", "ens4")   # ens3 = lgbm+xgb+cat, ens4 = hgb+lgbm+xgb+cat; *_big = larger, GPU when available


def threads() -> int:
    """CPU threads for the boosters: all the machine has (a free Colab has 2, a laptop 12), overridable with TRADINGBOT_THREADS."""
    import os
    try:
        return max(1, int(os.environ.get("TRADINGBOT_THREADS", "") or (os.cpu_count() or 1)))
    except ValueError:
        return max(1, os.cpu_count() or 1)


def gpu_available() -> bool:
    """True when an NVIDIA GPU is visible (nvidia-smi works). TRADINGBOT_GPU=0 forces the CPU, =1 forces the GPU path."""
    import os
    import shutil
    flag = os.environ.get("TRADINGBOT_GPU", "").lower()
    if flag in ("0", "false", "no"):
        return False
    if flag in ("1", "true", "yes"):
        return True
    return shutil.which("nvidia-smi") is not None


def _make_classifier(algo: str, seed: int, params: dict | None = None):
    """One classifier. lightgbm/xgboost/catboost are imported lazily (optional dependencies)."""
    if algo == "hgb":
        return HistGradientBoostingClassifier(**{**DEFAULT_PARAMS, **(params or {})}, random_state=seed)
    if algo == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=seed, n_jobs=threads())
    if algo == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
                             min_child_weight=50, tree_method="hist", n_jobs=threads(), random_state=seed, verbosity=0)
    if algo == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=5, l2_leaf_reg=5.0, verbose=0, random_seed=seed, thread_count=threads())
    if algo == "xgb_big":                  # bigger XGBoost: deeper, more trees; on the GPU when there is one (a T4 does this in seconds)
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0, min_child_weight=100,
                             tree_method="hist", device="cuda" if gpu_available() else "cpu", n_jobs=threads(), random_state=seed, verbosity=0)
    if algo == "cat_big":
        from catboost import CatBoostClassifier
        kw = dict(task_type="GPU") if gpu_available() else dict(thread_count=threads())
        return CatBoostClassifier(iterations=1000, learning_rate=0.04, depth=6, l2_leaf_reg=8.0, verbose=0, random_seed=seed, **kw)
    raise ValueError(f"unknown algo {algo}")


def _algo_list(algo: str) -> list[str]:
    return {"ens3": ["lgbm", "xgb", "cat"], "ens4": ["hgb", "lgbm", "xgb", "cat"]}.get(algo, [algo])


def train(ds: pd.DataFrame, asof: pd.Timestamp, window_years: int | None = 6, seed: int = 0,
          target: str = "label_xs", feature_set: str = "xs", params: dict | None = None, n_models: int = 1,
          algo: str = "hgb", date_col: str = "label_date"):
    """Fit using only rows whose label was known on `asof` (label_date <= asof).
    target='rank_target' fits a regressor on the cross-sectional rank of the forward return; otherwise a classifier.
    n_models > 1 averages several models with different seeds (use params max_features < 1 for diversity)."""
    tr = ds[(ds[date_col] <= asof) & ds[target].notna()]
    if window_years:
        tr = tr[tr["date"] >= asof - pd.DateOffset(years=window_years)]
    tr = tr.dropna(subset=["rsi14", "dist_ma50"])
    if len(tr) < 500:
        raise ValueError(f"only {len(tr)} training rows before {asof.date()}")
    cols = [c for c in feature_columns(feature_set) if c in tr and tr[c].notna().any()]  # e.g. no benchmark -> all-NaN rel-strength
    reg = target in ("rank_target", "excess_ret", "ret_fwd1") or algo in ("lgbm_reg", "lgbm_rank")
    y = tr[target] if reg else tr[target].astype(int)
    models = []
    if algo == "lgbm_rank":
        from lightgbm import LGBMRanker
        tr = tr.sort_values("date"); y = tr["rank_q"].astype(int); grp = tr.groupby("date", sort=True).size().to_numpy()
        for k in range(n_models):
            m = LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, random_state=seed + k, n_jobs=threads(), label_gain=[0, 1, 2, 3, 4])
            m.fit(tr[cols], y, group=grp); models.append(m)
    elif algo == "lgbm_reg":
        from lightgbm import LGBMRegressor
        for k in range(n_models):
            m = LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=5.0, verbose=-1, random_state=seed + k, n_jobs=threads())
            m.fit(tr[cols], y, ) ; models.append(m)
    elif reg:
        for k in range(n_models):
            m = HistGradientBoostingRegressor(**{**DEFAULT_PARAMS, **(params or {})}, random_state=seed + k)
            m.fit(tr[cols], y)
            models.append(m)
    else:
        for name in _algo_list(algo):
            for k in range(n_models):
                m = _make_classifier(name, seed + k, params if name == "hgb" else None)
                m.fit(tr[cols], y)
                models.append(m)
    ens = Ensemble(models, cols, reg)
    return ens


class Ensemble:
    """Average of one or more fitted models. For regressors the score is mapped to a (0,1) percentile-like value."""
    def __init__(self, models, cols, regression=False):
        self.models, self.cols_, self.regression = models, cols, regression

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if self.regression:
            return np.mean([m.predict(X[self.cols_]) for m in self.models], axis=0)
        return np.mean([m.predict_proba(X[self.cols_])[:, 1] for m in self.models], axis=0)


def predict(model, rows: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_score(rows), index=rows.index, name="p_up")
