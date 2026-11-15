"""Conviction sizing on v4's buys, 2024 validation only."""
import pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt"); a, b = "2024-01-02", "2024-12-31"
p1, p10 = F.recipe_signals(ds, F.default_recipes()["cross2r"], "2023-01-03", "2024-12-31", Path("runs/falcon_v31"))
p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in (p1, p10)]; sig = rank_signals(p1, p10, 30, 90, 0.7)
for sz in ("equal", "invvol", "score", "invvol_score"):
    s = S.simulate(sig, uni, a, b, S.real_world(1_000_000.0, max_pos=30, stop_atr=4.0, sizing=sz))["stats"]
    print({"sizing": sz, "ret": round(s["return"], 4), "sharpe": round(s["sharpe"], 2), "dd": round(s["max_drawdown"], 4)}, flush=True)
print("DONE", flush=True)
