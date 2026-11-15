"""Extra trading ideas for v4 on the cross2r predictions, scored on 2024 only (validation), strict real-world rules, TA-125 stocks only."""
import pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt")
r = F.default_recipes()["cross2r"]; a, b = "2024-01-02", "2024-12-31"
p1, p10 = F.recipe_signals(ds, r, "2023-01-03", "2024-12-31", Path("runs/falcon_v31"))
p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in (p1, p10)]
sig = rank_signals(p1, p10, 30, 90, 0.7)
ma = {w: bench.rolling(w).mean() for w in (50, 100, 200)}
variants = {"v4 as locked": {}}
for w in (50, 100, 200):
    on = (bench > ma[w]); variants[f"regime: index above {w}-day average (else no new buys)"] = dict(regime=on)
    variants[f"regime {w}d + sell everything when off"] = dict(regime=on, liquidate_on_risk_off=True)
for t in (3.0, 5.0): variants[f"trailing stop {t} ATR"] = dict(trail_atr=t)
for n in (10, 15): variants[f"only top {n} positions"] = dict(max_pos=n)
rows = []
for name, kw in variants.items():
    kw = dict(max_pos=30, stop_atr=4.0, sizing="invvol") | kw
    s = S.simulate(sig if kw["max_pos"] == 30 else rank_signals(p1, p10, kw["max_pos"], kw["max_pos"] * 3, 0.7), uni, a, b, S.real_world(1_000_000.0, **kw))["stats"]
    rows.append(dict(variant=name, ret=s["return"], sharpe=s["sharpe"], dd=s["max_drawdown"], trades=s["trades"])); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("runs/v4_ideas.csv", index=False); print("DONE", flush=True)
