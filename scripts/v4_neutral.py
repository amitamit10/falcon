"""Sector-neutralise the model's rankings AFTER prediction (no retraining): score = rank - lam x (the sector's mean rank that day). 2024 validation only."""
import pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
uni = None
ds, uni, bench = F.load_all("data/tase125.txt"); sec = dict(pd.read_csv("data/sectors.csv")[["ticker", "sector"]].values)
a, b = "2024-01-02", "2024-12-31"
p1, p10 = F.recipe_signals(ds, F.default_recipes()["cross2r"], "2023-01-03", "2024-12-31", Path("runs/falcon_v31"))
p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)].copy() for x in (p1, p10)]
def neutral(p, lam):
    p = p.copy(); p["_s"] = p["ticker"].map(sec).fillna("Unknown"); p["r"] = p.groupby("date")["p"].rank(pct=True)
    p["p"] = p["r"] - lam * p.groupby(["date", "_s"])["r"].transform("mean") + lam * 0.5; return p[["date", "ticker", "p"]]
rows = []
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    q1, q10 = (neutral(p1, lam), neutral(p10, lam)) if lam else (p1, p10)
    s = S.simulate(rank_signals(q1, q10, 30, 90, 0.7), uni, a, b, S.real_world(1_000_000.0, max_pos=30, stop_atr=4.0, sizing="invvol"))["stats"]
    rows.append(dict(lam=lam, ret=s["return"], sharpe=s["sharpe"], dd=s["max_drawdown"], trades=s["trades"])); print(rows[-1], flush=True)
print("DONE", flush=True)
