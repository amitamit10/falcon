"""Setups that professional chart traders screen for (Minervini / O'Neil style), as numbers a model can weigh. Daily bars only; the value at date t uses data up to and including t.
  vcp_atr_ratio     short vs longer true range (volatility contraction: small = tightening)
  vcp_range_contr   price range of the last 10 days over the last 30 (small = a tight base)
  vol_dryup         5-day average volume over the 50-day average (small = selling has dried up)
  brk_20d, brk_vol  closed above the prior 20-day high; and volume that day over the 50-day average (the breakout needs demand)
  rs_126            126-day return minus the market's (equal-weight universe) return: relative strength
  rs_line_near_high stock/market ratio versus its own 126-day high (0 = the relative-strength line is at a new high)
  above_52w_low     close over the 252-day low minus 1 (the "30% above the low" rule)
  up_down_vol       volume on up days over volume on down days, last 20 days (accumulation vs distribution)
  tight_close       standard deviation of the last 5 closes over the close
  trend_template    how many of 6 trend rules hold: above 150 and 200-day averages, 150 over 200, 200-day average rising, within 25% of the 52-week high, 30% over the low"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRO_COLUMNS = ["vcp_atr_ratio", "vcp_range_contr", "vol_dryup", "brk_20d", "brk_vol", "rs_126", "rs_line_near_high", "above_52w_low", "up_down_vol", "tight_close", "trend_template"]


def pro_features(df: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"].astype(float)
    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    f = pd.DataFrame(index=df.index)
    f["vcp_atr_ratio"] = tr.rolling(5).mean() / tr.rolling(30).mean()
    f["vcp_range_contr"] = (h.rolling(10).max() - lo.rolling(10).min()) / (h.rolling(30).max() - lo.rolling(30).min())
    f["vol_dryup"] = v.rolling(5).mean() / v.rolling(50).mean()
    brk = (c > h.shift().rolling(20).max()).astype(float).where(h.rolling(21).max().notna())
    f["brk_20d"] = brk; f["brk_vol"] = brk * v / v.rolling(50).mean()
    m = mkt.reindex(df.index).ffill()
    f["rs_126"] = c.pct_change(126) - m.pct_change(126)
    rs = c / m; f["rs_line_near_high"] = rs / rs.rolling(126, min_periods=60).max() - 1
    lo252, hi252 = lo.rolling(252, min_periods=120).min(), h.rolling(252, min_periods=120).max()
    f["above_52w_low"] = c / lo252 - 1
    up = v.where(c > c.shift(), 0.0).rolling(20).sum(); dn = v.where(c < c.shift(), 0.0).rolling(20).sum()
    f["up_down_vol"] = up / dn.replace(0, np.nan)
    f["tight_close"] = c.rolling(5).std() / c
    ma150, ma200 = c.rolling(150).mean(), c.rolling(200).mean()
    rules = [c > ma150, c > ma200, ma150 > ma200, ma200 > ma200.shift(20), c >= 0.75 * hi252, c >= 1.3 * lo252]
    f["trend_template"] = sum(r.astype(float) for r in rules).where(ma200.notna() & hi252.notna())
    return f


def add_pro(ds: pd.DataFrame, universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = pd.concat({tk: df["close"] for tk, df in universe.items()}, axis=1).sort_index()
    mkt = (1 + closes.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()           # equal-weight market proxy, point-in-time
    rows = []
    for tk, df in universe.items():
        f = pro_features(df, mkt); f["ticker"] = tk; f.index.name = "date"; rows.append(f.reset_index())
    return ds.merge(pd.concat(rows, ignore_index=True), on=["date", "ticker"], how="left")
