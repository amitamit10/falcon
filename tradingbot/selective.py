"""Selective mode: act only on the few calls the model is most sure about.

BUY candidate: at least one question is flagged high-confidence UP (calibrated probability at or above the validated bar) and neither is flagged DOWN.
  Hold period: 10 trading days when the 10-day question flagged it, otherwise 1 day. Score = mean calibrated probability of the flagged-up questions
  (+0.02 when both agree), used to rank candidates when there are more than free slots.
SELL (exit) signal: any question flagged high-confidence DOWN for a stock we hold.
Everything here uses only the flags known at the close of the signal day; the simulator and the daily report share this one rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_row(p1, f1, p10, f10) -> tuple[bool, bool, float, int]:
    """(buy, sell, score, hold_days) for one stock on one day."""
    up1, up10 = f1 == "up", f10 == "up"
    down = f1 == "down" or f10 == "down"
    buy = (up1 or up10) and not down
    ps = [p for p, u in ((p1, up1), (p10, up10)) if u]
    score = (float(np.mean(ps)) + (0.02 if up1 and up10 else 0.0)) if ps else 0.0
    return buy, down, score, 10 if up10 else 1


def signals_from_predictions(p1: pd.DataFrame, p10: pd.DataFrame, cal1, cal10) -> pd.DataFrame:
    """p1 / p10: frames with date, ticker, p (raw ensemble probability). Returns date, ticker, p1, p10, f1, f10, buy, sell, score, hold (calibrated flags)."""
    a = p1[["date", "ticker", "p"]].rename(columns={"p": "r1"}); b = p10[["date", "ticker", "p"]].rename(columns={"p": "r10"})
    m = a.merge(b, on=["date", "ticker"], how="inner")
    m["p1"], m["p10"] = cal1.apply(m["r1"].to_numpy()), cal10.apply(m["r10"].to_numpy())
    m["f1"], m["f10"] = cal1.flag(m["p1"].to_numpy()), cal10.flag(m["p10"].to_numpy())
    rows = [signal_row(*x) for x in zip(m["p1"], m["f1"], m["p10"], m["f10"])]
    m["buy"], m["sell"], m["score"], m["hold"] = zip(*rows) if rows else ([], [], [], [])
    return m


def picks(scan: pd.DataFrame, max_n: int = 5) -> pd.DataFrame:
    """Today's selective picks from the daily scan (needs p1, p10, flag1, flag10): the best BUY candidates and the stocks flagged for exit."""
    if "flag1" not in scan:
        return scan.iloc[0:0]
    s = scan.copy()
    r = [signal_row(a, b, c, d) for a, b, c, d in zip(s["p1"], s["flag1"], s["p10"], s["flag10"])]
    s["buy"], s["sell"], s["score"], s["hold"] = zip(*r) if r else ([], [], [], [])
    buys = s[s["buy"]].sort_values("score", ascending=False).head(max_n)
    return buys.assign(action="BUY")


def rank_signals(p1: pd.DataFrame, p10: pd.DataFrame, top_n: int = 10, drop_rank: int = 30, w10: float = 0.7) -> pd.DataFrame:
    """Daily ranking policy: every day rank all stocks by a blend of the two model scores (w10 on the 10-day question, the rest on the next-day question).
    BUY a stock that enters the top `top_n`; keep (hold) it while it stays inside `drop_rank`; SELL when it falls below `drop_rank`. The wide gap between the two
    ranks stops it from trading in and out on small changes (every fill costs money). Output has the same columns the simulator reads."""
    a = p1[["date", "ticker", "p"]].rename(columns={"p": "r1"}); b = p10[["date", "ticker", "p"]].rename(columns={"p": "r10"})
    m = a.merge(b, on=["date", "ticker"], how="inner")
    g = m.groupby("date")
    m["score"] = w10 * g["r10"].rank(pct=True) + (1 - w10) * g["r1"].rank(pct=True)
    m["rank"] = m.groupby("date")["score"].rank(ascending=False, method="first")
    m["buy"] = m["rank"] <= top_n
    m["sell"] = m["rank"] > drop_rank
    m["hold"] = 10 ** 6
    return m
