"""Sector and cross-asset flow from the US session. The US session of day t closes after the TASE close, so by the next morning its sector moves are known:
a US bank ETF falling 2% overnight says something about TASE banks that the TASE close cannot. Each stock is mapped (data/sectors.csv, Yahoo's sector) to the US ETF of
its sector; market-wide factors are gold, the dollar, credit, long bonds and EUR/ILS. All values at date t are the US-session returns of date t (known before the next TASE open).
Per stock:  sector_us_ret_1d, sector_us_ret_5d, sector_us_rel_1d (sector ETF return minus the S&P 500's).
Market:     gold_ret_1d, usd_ret_1d, credit_ret_1d, tlt_ret_1d, eurils_ret_1d.   Missing data stays NaN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .crosslist import EMPTY, _close

SECTOR_ETF = {"Financial Services": "XLF", "Technology": "XLK", "Healthcare": "XLV", "Real Estate": "VNQ", "Energy": "XLE", "Industrials": "XLI", "Utilities": "XLU",
              "Basic Materials": "XLB", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Communication Services": "XLC"}
MARKET_ETF = {"gold_ret_1d": "GLD", "usd_ret_1d": "UUP", "credit_ret_1d": "HYG", "tlt_ret_1d": "TLT", "eurils_ret_1d": "EURILS=X"}
SECTOR_COLUMNS = ["sector_us_ret_1d", "sector_us_ret_5d", "sector_us_rel_1d"] + list(MARKET_ETF)


def sector_flow(universe: dict, sectors: pd.DataFrame | None, series: dict[str, pd.Series] | None = None, spx: pd.Series | None = None) -> pd.DataFrame:
    """Long frame date, ticker + SECTOR_COLUMNS. `series` (ETF ticker -> daily closes) and `spx` can be supplied (tests, offline); otherwise they are downloaded."""
    smap = dict(zip(sectors["ticker"], sectors["sector"])) if sectors is not None else {}
    need = set(SECTOR_ETF.values()) | set(MARKET_ETF.values())
    series = {t: _close(t) for t in need} if series is None else series
    spx = _close("^GSPC") if spx is None else spx
    dates = sorted({d for df in universe.values() for d in df.index})
    idx = pd.DatetimeIndex(dates)
    r1 = {t: s.pct_change().reindex(idx) for t, s in series.items() if len(s)}
    r5 = {t: s.pct_change(5).reindex(idx) for t, s in series.items() if len(s)}
    sp1 = spx.pct_change().reindex(idx) if len(spx) else pd.Series(np.nan, index=idx)
    mk = pd.DataFrame({col: r1.get(t, pd.Series(np.nan, index=idx)) for col, t in MARKET_ETF.items()}, index=idx)
    rows = []
    for tk, df in universe.items():
        etf = SECTOR_ETF.get(smap.get(tk, ""), "")
        f = pd.DataFrame(index=df.index)
        f["sector_us_ret_1d"] = r1[etf].reindex(df.index) if etf in r1 else np.nan
        f["sector_us_ret_5d"] = r5[etf].reindex(df.index) if etf in r5 else np.nan
        f["sector_us_rel_1d"] = f["sector_us_ret_1d"] - sp1.reindex(df.index)
        for c in MARKET_ETF:
            f[c] = mk[c].reindex(df.index)
        f["ticker"] = tk; f.index.name = "date"; rows.append(f.reset_index())
    return pd.concat(rows, ignore_index=True)[["date", "ticker"] + SECTOR_COLUMNS]


def add_sector_flow(ds: pd.DataFrame, universe: dict, sectors_path: str | Path = "data/sectors.csv", **kw) -> pd.DataFrame:
    p = Path(sectors_path)
    return ds.merge(sector_flow(universe, pd.read_csv(p) if p.exists() else None, **kw), on=["date", "ticker"], how="left")
