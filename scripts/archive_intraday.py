"""Appends the latest 5-minute bars (Yahoo keeps only ~60 days) of every TA-125 stock to a growing archive, so minute-level tests become possible later.
Run it at least every few weeks (daily is fine); it merges and de-duplicates. Archive: data/intraday_archive/<ticker>_5m.csv.gz"""
import sys, pandas as pd
from pathlib import Path
import yfinance as yf
from tradingbot.data import load_tickers
TZ = "Asia/Jerusalem"; out = Path("data/intraday_archive"); out.mkdir(parents=True, exist_ok=True)
tickers = load_tickers(sys.argv[1] if len(sys.argv) > 1 else "data/tase125.txt"); added = 0
for tk in tickers:
    try:
        d = yf.download(tk, period="60d", interval="5m", progress=False, auto_adjust=False)
        if d is None or d.empty: continue
        d.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in d.columns]
        d = d[["open", "high", "low", "close", "volume"]].dropna(); d = d[d["close"] > 0]
        d.index = d.index.tz_convert(TZ) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(TZ)
        f = out / f"{tk}_5m.csv.gz"
        if f.exists():
            old = pd.read_csv(f, index_col=0); old.index = pd.to_datetime(old.index, utc=True).tz_convert(TZ); d = pd.concat([old, d])
        d = d[~d.index.duplicated(keep="last")].sort_index(); d.to_csv(f); added += 1
    except Exception as e:
        print("skip", tk, e)
print(f"archived {added}/{len(tickers)} stocks into {out}")
