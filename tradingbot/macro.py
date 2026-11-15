"""Overnight / cross-market features for the next-day model. For TASE date t the US session of date t closes after the TASE close (23:00 vs 17:25 Israel time),
so it is known by the morning of t+1, before the TASE opens: legitimate for a morning prediction of the move from close t to close t+1.
Series: S&P 500, Nasdaq, VIX, USD/ILS, Brent oil, US 10y yield. Missing days (a US holiday) stay NaN."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

SERIES = {"gspc": "^GSPC", "ndx": "^IXIC", "vix": "^VIX", "ils": "ILS=X", "oil": "BZ=F", "tnx": "^TNX"}
MACRO_COLUMNS = ["us_ret_1d", "us_ret_5d", "ndx_ret_1d", "vix_level", "vix_chg_1d", "ils_ret_1d", "ils_ret_5d", "oil_ret_1d", "tnx_chg_1d"]
CACHE = Path(os.environ.get("TRADINGBOT_MACRO_CACHE", "data/macro_cache"))


MAX_AGE_HOURS = 6          # a cached series older than this is refreshed (the morning run needs yesterday's US close)


def _close(t: str, start: str, refresh: bool) -> pd.Series:
    """Daily closes of one series, cached on disk. Falls back to the cache when offline, and to an empty series (NaN features) when there is neither."""
    import time
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (t.replace("^", "") + ".csv")
    fresh = p.exists() and (time.time() - p.stat().st_mtime) < MAX_AGE_HOURS * 3600
    if p.exists() and fresh and not refresh:
        return pd.read_csv(p, index_col=0, parse_dates=True)["close"]
    try:
        d = yf.download(t, start=start, auto_adjust=True, progress=False)
        c = d["Close"].iloc[:, 0] if hasattr(d["Close"], "columns") else d["Close"]
        s = c.dropna().rename("close"); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        if len(s):
            s.to_frame().to_csv(p)
            return s
    except Exception:  # noqa: BLE001 - offline or a Yahoo hiccup: use what we have
        pass
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)["close"]
    return pd.Series(dtype=float, name="close", index=pd.DatetimeIndex([]))


def macro_frame(start: str = "2005-01-01", refresh: bool = False) -> pd.DataFrame:
    """One row per calendar date with the macro columns (each uses only that date and earlier)."""
    c = {k: _close(t, start, refresh) for k, t in SERIES.items()}
    out = pd.DataFrame(index=sorted(set().union(*[s.index for s in c.values()])), columns=[])
    out["us_ret_1d"] = c["gspc"].pct_change().reindex(out.index)
    out["us_ret_5d"] = c["gspc"].pct_change(5).reindex(out.index)
    out["ndx_ret_1d"] = c["ndx"].pct_change().reindex(out.index)
    out["vix_level"] = c["vix"].reindex(out.index)
    out["vix_chg_1d"] = c["vix"].pct_change().reindex(out.index)
    out["ils_ret_1d"] = c["ils"].pct_change().reindex(out.index)
    out["ils_ret_5d"] = c["ils"].pct_change(5).reindex(out.index)
    out["oil_ret_1d"] = c["oil"].pct_change().reindex(out.index)
    out["tnx_chg_1d"] = c["tnx"].diff().reindex(out.index)
    out.index.name = "date"
    return out


def add_macro(ds: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    return ds.merge(macro.reset_index(), on="date", how="left")
