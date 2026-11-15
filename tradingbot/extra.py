"""More ways of reading the stock data, all point-in-time (value at t uses bars <= t):
- candle shape of the day: gap at the open, range, where the close sits in the range, body
- streak of consecutive up/down closes, short/long realised-volatility ratio, volume shock vs the 20-day median
- stretch from the 20-day average (z-score of the Bollinger band), band-width percentile (squeeze)
- calendar: weekday, month, day of month
- sector context: the sector's own 1/5/20-day return and breadth, and the stock relative to its sector
- cross-sectional ranks of the stretch, the day's return, volume shock and range"""
from __future__ import annotations

import numpy as np
import pandas as pd

STOCK_COLUMNS = ["gap_open", "range_pct", "close_in_range", "body_pct", "streak", "rv5_20", "vol_shock", "z_bb20", "bw_pctile"]
CAL_COLUMNS = ["dow", "month", "dom"]
SECTOR_COLUMNS = ["sector_ret_1d", "sector_ret_5d", "sector_ret_20d", "sector_breadth50", "rel_sector_1d", "rel_sector_5d", "rel_sector_20d"]
XS_EXTRA = ["xs_z_bb20", "xs_ret_1d", "xs_vol_shock", "xs_range_pct", "xs_gap_open", "ret_1d_resid"]
EXTRA_COLUMNS = STOCK_COLUMNS + CAL_COLUMNS + SECTOR_COLUMNS + XS_EXTRA


def stock_extra(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l, v = (df[k].astype(float) for k in ("close", "open", "high", "low", "volume"))
    ret = c.pct_change(); rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    f["gap_open"] = o / c.shift(1) - 1
    f["range_pct"] = (h - l) / c
    f["close_in_range"] = (c - l) / rng
    f["body_pct"] = (c - o) / o
    sign = np.sign(ret.fillna(0)); grp = (sign != sign.shift()).cumsum()
    f["streak"] = sign * (sign.groupby(grp).cumcount() + 1)
    f["rv5_20"] = ret.rolling(5).std() / ret.rolling(20).std().replace(0, np.nan)
    f["vol_shock"] = v / v.rolling(20).median().replace(0, np.nan)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["z_bb20"] = (c - ma20) / sd20.replace(0, np.nan)
    f["bw_pctile"] = (4 * sd20 / ma20).rolling(120).rank(pct=True)
    f.index.name = "date"
    return f


def add_extra(ds: pd.DataFrame, universe: dict[str, pd.DataFrame], sectors: pd.DataFrame | None, min_sector: int = 4) -> pd.DataFrame:
    parts = []
    for tk in ds["ticker"].unique():
        if tk in universe and len(universe[tk]) > 60:
            e = stock_extra(universe[tk]).reset_index(); e["ticker"] = tk; parts.append(e)
    out = ds.merge(pd.concat(parts), on=["date", "ticker"], how="left")
    d = pd.to_datetime(out["date"])
    out["dow"], out["month"], out["dom"] = d.dt.dayofweek, d.dt.month, d.dt.day
    sec = dict(zip(sectors["ticker"], sectors["sector"])) if sectors is not None else {}
    out["_sector"] = out["ticker"].map(sec).fillna("Unknown")
    g = out.groupby(["date", "_sector"])
    n = g["ticker"].transform("count")
    for k, col in (("1d", "ret_1d"), ("5d", "ret_5d"), ("20d", "ret_20d")):
        sm = g[col].transform("mean").where(n >= min_sector)                # a sector of 1-3 stocks says nothing beyond the stock itself
        out[f"sector_ret_{k}"] = sm; out[f"rel_sector_{k}"] = out[col] - sm
    out["sector_breadth50"] = (out["dist_ma50"] > 0).astype(float).where(out["dist_ma50"].notna()).groupby([out["date"], out["_sector"]]).transform("mean").where(n >= min_sector)
    gd = out.groupby("date")
    for new, src in (("xs_z_bb20", "z_bb20"), ("xs_ret_1d", "ret_1d"), ("xs_vol_shock", "vol_shock"), ("xs_range_pct", "range_pct"), ("xs_gap_open", "gap_open")):
        out[new] = gd[src].rank(pct=True)
    out["ret_1d_resid"] = out["ret_1d"] - gd["ret_1d"].transform("mean")
    return out.drop(columns="_sector")
