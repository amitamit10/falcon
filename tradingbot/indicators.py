"""Plain pandas/numpy indicators (no look-ahead: value at t uses data <= t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna() & dn.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    lo = df["low"].rolling(k, min_periods=k).min()
    hi = df["high"].rolling(k, min_periods=k).max()
    pk = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pk, pk.rolling(d, min_periods=d).mean()


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    md = tp.rolling(n, min_periods=n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def vwma(close: pd.Series, volume: pd.Series, n: int = 20) -> pd.Series:
    return (close * volume).rolling(n, min_periods=n).sum() / volume.rolling(n, min_periods=n).sum()
