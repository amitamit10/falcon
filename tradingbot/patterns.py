"""Classic chart patterns, all point-in-time (value at t uses bars <= t only):
trend lines through confirmed pivot lows, Fibonacci retracement of the latest up-leg, cup-and-handle with
the usual price target (breakout + cup depth), and unfilled price gaps (gaps tend to close)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def trendline(df: pd.DataFrame, k: int = 5, max_age: int = 120) -> pd.DataFrame:
    """Rising support line through the last two CONFIRMED pivot lows (the later low must be higher).
    Returns line value at t, slope per bar, and distance of close to the line."""
    lo = df["low"].to_numpy(float); c = df["close"].to_numpy(float); n = len(df)
    roll = pd.Series(lo).rolling(2 * k + 1, center=True).min().to_numpy()
    piv = [(i, lo[i]) for i in range(k, n - k) if lo[i] == roll[i]]   # pivot at i is confirmed at i + k
    line, slope = np.full(n, np.nan), np.full(n, np.nan)
    j = 0
    for t in range(n):
        known = [p for p in piv if p[0] + k <= t]
        if len(known) >= 2:
            (i1, v1), (i2, v2) = known[-2], known[-1]
            if v2 > v1 and i2 > i1 and t - i2 <= max_age:
                m = (v2 - v1) / (i2 - i1)
                line[t], slope[t] = v2 + m * (t - i2), m
    out = pd.DataFrame({"trendline": line, "tl_slope": slope / c}, index=df.index)
    out["dist_trendline"] = df["close"] / out["trendline"] - 1
    return out


def fibonacci(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Retracement of the latest up-leg inside the lookback window: swing low BEFORE swing high.
    fib_retr = fraction of the leg given back (0 = at the high, 1 = back at the low). NaN if no up-leg."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n, retr = len(df), np.full(len(df), np.nan)
    for t in range(lookback - 1, n):
        hw, lw = h[t - lookback + 1:t + 1], l[t - lookback + 1:t + 1]
        ih = int(np.argmax(hw)); il = int(np.argmin(lw[:ih + 1])) if ih > 0 else 0
        leg = hw[ih] - lw[il]
        if ih > il and leg > 0 and leg / hw[ih] > 0.08:
            retr[t] = (hw[ih] - c[t]) / leg
    out = pd.DataFrame({"fib_retr": retr}, index=df.index)
    out["fib_zone"] = out["fib_retr"].between(0.382, 0.618).astype(float).where(out["fib_retr"].notna())
    return out


def cup_handle(df: pd.DataFrame, lookback: int = 90, min_depth: float = 0.12, max_depth: float = 0.40) -> pd.DataFrame:
    """Cup and handle: left rim -> rounded bottom (12-40% deep) -> right rim near the left rim -> short shallow handle.
    cup_breakout = close crosses above the rim; cup_target = rim + cup depth (the textbook target: a rise by the cup depth)."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(df); brk, tgt = np.zeros(n), np.full(n, np.nan)
    for t in range(lookback, n):
        w0 = t - lookback
        hw, lw = h[w0:t], l[w0:t]                      # bars before t (t is the candidate breakout bar)
        il = int(np.argmax(hw[: lookback // 2]))       # left rim in the first half
        rim = hw[il]
        ib = il + int(np.argmin(lw[il:]))              # cup bottom after the left rim
        depth = (rim - lw[ib]) / rim
        if not (min_depth <= depth <= max_depth) or ib - il < 15 or ib > lookback - 8:
            continue
        right = hw[ib:]; ir = int(np.argmax(right))
        if right[ir] < rim * 0.95 or right[ir] > rim * 1.03:       # right rim within ~5% of the left rim
            continue
        handle = lw[ib + ir:]
        if len(handle) < 4 or len(handle) > 30 or (right[ir] - handle.min()) > 0.5 * (rim - lw[ib]):
            continue                                              # handle: short, gives back < half the cup
        top = max(rim, right[ir])
        if c[t] > top and c[t - 1] <= top:
            brk[t], tgt[t] = 1.0, top + (rim - lw[ib])
    return pd.DataFrame({"cup_breakout": brk, "cup_target": tgt}, index=df.index)


def gaps(df: pd.DataFrame, max_age: int = 60) -> pd.DataFrame:
    """Most recent UNFILLED gap. gap_down_target = how far above price the gap-down zone's top sits (price tends to
    return to fill it); gap_up_support = how far below price an unfilled gap-up's bottom sits."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(df); gd, gu = np.full(n, np.nan), np.full(n, np.nan)
    down, up = None, None                      # (bar index, level that must be touched to fill)
    for t in range(1, n):
        if o[t] < l[t - 1] * 0.995:            # gap down: filled when price trades back up to the prior low
            down = (t, l[t - 1])
        if o[t] > h[t - 1] * 1.005:            # gap up: filled when price trades back down to the prior high
            up = (t, h[t - 1])
        if down and h[t] >= down[1] and t > down[0]: down = None
        if up and l[t] <= up[1] and t > up[0]: up = None
        if down and t - down[0] <= max_age: gd[t] = down[1] / c[t] - 1
        if up and t - up[0] <= max_age: gu[t] = c[t] / up[1] - 1
    return pd.DataFrame({"gap_down_target": gd, "gap_up_support": gu}, index=df.index)


def all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([trendline(df), fibonacci(df), cup_handle(df), gaps(df)], axis=1)


def claims(universe: dict[str, pd.DataFrame], start: str, end: str, gap_min: float = 0.01) -> dict:
    """Event studies of two popular chart-pattern claims: (1) most gaps get filled, (2) a cup-and-handle rises by its depth.
    Forward windows are looked at AFTER the event only to score it; nothing here feeds a signal."""
    gap_events = {20: [], 60: []}; cup_hits, cup_ret20, cup_n = [], [], 0
    for tk, df in universe.items():
        if len(df) < 300:
            continue
        o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        ix = df.index
        for t in range(1, len(df) - 1):
            if not (start <= str(ix[t].date()) <= end):
                continue
            for lvl, hit in ((l[t - 1], o[t] < l[t - 1] * (1 - gap_min)), (h[t - 1], o[t] > h[t - 1] * (1 + gap_min))):
                if not hit:
                    continue
                down = o[t] < lvl
                for horizon in gap_events:
                    seg = slice(t, min(t + horizon, len(df)))
                    gap_events[horizon].append(bool((h[seg] >= lvl).any()) if down else bool((l[seg] <= lvl).any()))
        cp = cup_handle(df)
        for t in np.flatnonzero(cp["cup_breakout"].to_numpy() == 1.0):
            if not (start <= str(ix[t].date()) <= end) or t + 60 >= len(df):
                continue
            cup_n += 1
            cup_hits.append(bool((h[t + 1:t + 61] >= cp["cup_target"].iloc[t]).any()))
            cup_ret20.append(c[t + 20] / c[t] - 1)
    r = lambda a: round(float(np.mean(a)), 4) if a else None
    return {"gap_events": len(gap_events[20]), "gap_filled_within_20d": r(gap_events[20]),
            "gap_filled_within_60d": r(gap_events[60]), "cup_breakouts": cup_n,
            "cup_target_reached_within_60d": r(cup_hits), "cup_median_ret_20d": round(float(np.median(cup_ret20)), 4) if cup_ret20 else None}
