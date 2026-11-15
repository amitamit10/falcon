"""The extra inputs of the prediction models, added to the base dataset: overnight / cross-market data (macro.py), extra analysis features and sector context
(extra.py, data/sectors.csv) the Alpha158 family (alpha158.py) and the cross-listing signals (crosslist.py: US prices of dual-listed companies, EIS, semiconductors). Everything is point-in-time. A missing piece (no sector file, offline macro data) just leaves
its columns empty; the models drop all-empty columns, so the scan still runs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .alpha158 import add_alpha
from .crosslist import add_cross
from .extra import add_extra
from .fundhist import add_fund
from .macro import add_macro, macro_frame
from .sectorflow import add_sector_flow


def augment(ds: pd.DataFrame, universe: dict[str, pd.DataFrame], sectors_path: str | Path = "data/sectors.csv", refresh_macro: bool = False, fund_hist=None) -> pd.DataFrame:
    sp = Path(sectors_path)
    sectors = pd.read_csv(sp) if sp.exists() else None
    ds = add_macro(ds, macro_frame(refresh=refresh_macro))
    ds = add_extra(ds, universe, sectors)
    ds = add_alpha(ds, universe)
    ds = add_cross(ds, universe)
    ds = add_sector_flow(ds, universe, sectors_path=sectors_path)
    from .pro import add_pro
    ds = add_pro(ds, universe)
    ds = add_gauss_targets(ds, sectors)
    return add_fund(ds, universe, hist=fund_hist) if fund_hist is not False else ds


def add_gauss_targets(ds: pd.DataFrame, sectors: pd.DataFrame | None) -> pd.DataFrame:
    """Training targets (they use the future, so only rows whose label date is known may be trained on, as for every other target):
    rk_gauss = the day's cross-sectional rank of the 10-day forward return pushed through the normal quantile function (no outliers, the same scale every day);
    sn_gauss = the same after taking out each sector's average rank that day (sector-neutral: the model must pick the best stock inside its sector)."""
    from scipy.special import ndtri
    r = ds.groupby("date")["next_ret"].rank(pct=True)
    n = ds.groupby("date")["next_ret"].transform("count").clip(lower=2)
    ds["rk_gauss"] = ndtri(((r * n - 0.5) / n).clip(1e-4, 1 - 1e-4))
    sec = ds["ticker"].map(dict(zip(sectors["ticker"], sectors["sector"]))).fillna("Unknown") if sectors is not None else pd.Series("Unknown", index=ds.index)
    rs = r - r.groupby([ds["date"], sec]).transform("mean") + 0.5
    ds["sn_gauss"] = ndtri(rs.clip(1e-4, 1 - 1e-4))
    return ds
