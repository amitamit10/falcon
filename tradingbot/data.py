"""Price loading with an on-disk CSV cache. Yahoo Finance by default; any CSV dir works offline."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE = Path(os.environ.get("TRADINGBOT_CACHE", "data/cache"))      # a Colab run points this at Google Drive so downloads survive a disconnect
COLS = ["open", "high", "low", "close", "volume"]


def load_tickers(path: str | Path = "data/tase125.txt") -> list[str]:
    return [t.strip() for t in Path(path).read_text().splitlines() if t.strip() and not t.startswith("#")]


def fix_units(df: pd.DataFrame, lo: float = 60.0, hi: float = 160.0) -> pd.DataFrame:
    """Yahoo sometimes returns Tel Aviv history in mixed units: shekels (ILS) for older bars and agorot (ILA, x100)
    for newer ones, or the reverse. A one-bar close ratio of ~100x (or ~1/100) is a unit switch, not a real move, so rescale
    everything BEFORE the switch to the unit of the latest data. Also repairs one-tick OHLC glitches (a low or high far
    outside the open/close range). Volume is left unchanged (it is a share count)."""
    d = df.copy()
    for _ in range(10):                                   # a series can switch units more than once
        r = d["close"] / d["close"].shift(1)
        up = (r >= lo) & (r <= hi)                        # older bars were in the smaller unit
        down = (r <= 1 / lo) & (r >= 1 / hi)              # older bars were in the larger unit
        sw = r[up | down]
        if sw.empty:
            break
        i = d.index.get_loc(sw.index[-1])                 # latest switch first; earlier ones fixed on later passes
        factor = float(round(r.iloc[i]))if r.iloc[i] >= 1 else 1.0 / round(1.0 / r.iloc[i])
        for c in ("open", "high", "low", "close"):
            d.iloc[:i, d.columns.get_loc(c)] = d[c].iloc[:i] * factor
    body_lo, body_hi = d[["open", "close"]].min(axis=1), d[["open", "close"]].max(axis=1)
    d["low"] = d["low"].where(d["low"] >= 0.5 * body_lo, body_lo)
    d["high"] = d["high"].where(d["high"] <= 2.0 * body_hi, body_hi)
    return d


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[COLS].dropna(subset=["close"])
    return fix_units(df[~df.index.duplicated(keep="last")].sort_index())


def fetch_yahoo(ticker: str, start: str = "2014-01-01", end: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no data for {ticker}")
    return _normalize(df)


def get_prices(ticker: str, start: str = "2014-01-01", refresh: bool = False, cache: Path = CACHE) -> pd.DataFrame:
    """Cached daily OHLCV. A CSV dropped in the cache dir as <TICKER>.csv is used as-is (offline mode)."""
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{ticker}.csv"
    if p.exists() and not refresh:
        return _normalize(pd.read_csv(p, index_col=0, parse_dates=True))
    df = fetch_yahoo(ticker, start)
    df.to_csv(p)
    return df


def get_universe(tickers: list[str], **kw) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        try:
            out[t] = get_prices(t, **kw)
        except Exception as e:  # keep scanning on single-ticker failure
            print(f"skip {t}: {e}")
    return out
