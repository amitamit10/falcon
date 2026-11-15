"""Cross-listing signals. About a dozen TA-125 companies are also listed in the US (ESLT, TEVA, NICE, TSEM, NVMI, CAMT, ICL, OPK, NYAX, ORA, KEN, FORTY).
The US session of day t closes AFTER the TASE close, so by the next morning the US price has already moved: the gap between the US price (converted to agorot at USD/ILS)
and the TASE close says where the TASE stock is likely to open. Features per stock-day (known the morning after day t):
  dual_ret_1d : US-session return of the same company on date t
  dual_gap    : US price x USD/ILS x 100 / TASE close - 1 (positive: the US price is above the TASE price)
Market-level (same value for every stock): EIS (iShares MSCI Israel ETF, trades in US hours) and the semiconductor index return, and the mean gap / return of the dual-listed names.
Confirmed pairs are in data/dual_listed.csv (price ratio ~1 and correlated returns). Missing data stays NaN."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("TRADINGBOT_MACRO_CACHE", "data/macro_cache"))
MAX_AGE_HOURS = 6
CROSS_COLUMNS = ["dual_ret_1d", "dual_gap", "dual_gap_mean", "dual_ret_mean", "eis_ret_1d", "sox_ret_1d"]
EMPTY = pd.Series(dtype=float, name="close", index=pd.DatetimeIndex([]))


def _close(t: str, start: str = "2014-01-01") -> pd.Series:
    """Daily close of a US ticker, cached; offline falls back to the cache, then to an empty series."""
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / ("x_" + t.replace("^", "").replace("=", "_") + ".csv")
    if p.exists() and (time.time() - p.stat().st_mtime) < MAX_AGE_HOURS * 3600:
        return pd.read_csv(p, index_col=0, parse_dates=True)["close"]
    try:
        d = yf.download(t, start=start, auto_adjust=True, progress=False)
        c = d["Close"].iloc[:, 0] if hasattr(d["Close"], "columns") else d["Close"]
        s = c.dropna().rename("close"); s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
        if len(s):
            s.to_frame().to_csv(p); return s
    except Exception:  # noqa: BLE001
        pass
    return pd.read_csv(p, index_col=0, parse_dates=True)["close"] if p.exists() else EMPTY


def load_pairs(path: str | Path = "data/dual_listed.csv") -> dict[str, str]:
    p = Path(path)
    return dict(zip(*pd.read_csv(p)[["tase", "us"]].T.values)) if p.exists() else {}


def cross_features(universe: dict[str, pd.DataFrame], pairs: dict[str, str] | None = None, fx: pd.Series | None = None, us: dict[str, pd.Series] | None = None,
                   mkt: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Long frame date, ticker + CROSS_COLUMNS. fx / us / mkt can be passed in (tests, offline); otherwise they are downloaded."""
    pairs = load_pairs() if pairs is None else pairs
    fx = _close("ILS=X") if fx is None else fx
    us = {u: _close(u) for u in set(pairs.values())} if us is None else us
    mkt = {"eis": _close("EIS"), "sox": _close("^SOX")} if mkt is None else mkt
    rows = []
    for tk, df in universe.items():
        ta = df["close"].astype(float)
        f = pd.DataFrame(index=ta.index, columns=["dual_ret_1d", "dual_gap"], dtype=float)
        u = us.get(pairs.get(tk, ""), EMPTY)
        if len(u) and len(fx):
            usd = u.reindex(ta.index); rate = fx.reindex(ta.index).ffill()
            f["dual_ret_1d"] = u.pct_change().reindex(ta.index)
            f["dual_gap"] = usd * rate * 100 / ta - 1
        f["ticker"] = tk; f.index.name = "date"; rows.append(f.reset_index())
    out = pd.concat(rows, ignore_index=True)
    g = out.groupby("date")
    out["dual_gap_mean"] = g["dual_gap"].transform("mean"); out["dual_ret_mean"] = g["dual_ret_1d"].transform("mean")
    for k, col in (("eis", "eis_ret_1d"), ("sox", "sox_ret_1d")):
        s = mkt.get(k, EMPTY); out[col] = out["date"].map(s.pct_change()) if len(s) else np.nan
    return out[["date", "ticker"] + CROSS_COLUMNS]


def add_cross(ds: pd.DataFrame, universe: dict[str, pd.DataFrame], **kw) -> pd.DataFrame:
    return ds.merge(cross_features(universe, **kw), on=["date", "ticker"], how="left")
