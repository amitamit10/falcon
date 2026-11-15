"""Plain-code rule guards on v4's buys, each scored on 2024 only (validation), strict real-world rules."""
import pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.rules import apply_rules
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt"); sectors = pd.read_csv("data/sectors.csv")
a, b = "2024-01-02", "2024-12-31"
p1, p10 = F.recipe_signals(ds, F.default_recipes()["cross2r"], "2023-01-03", "2024-12-31", Path("runs/falcon_v31"))
p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in (p1, p10)]
sig = rank_signals(p1, p10, 30, 90, 0.7)
V = {"none (v4 as locked)": ([], 0, 0), "no_chase": (["no_chase"], 0, 0), "no_overbought": (["no_overbought"], 0, 0), "no_falling_knife": (["no_falling_knife"], 0, 0), "volume_ok": (["volume_ok"], 0, 0),
     "sector cap 3": ([], 3, 0), "sector cap 5": ([], 5, 0), "cooldown 5 days": ([], 0, 5), "cooldown 10 days": ([], 0, 10),
     "all four rules": (["no_chase", "no_overbought", "no_falling_knife", "volume_ok"], 0, 0),
     "all four + sector cap 5 + cooldown 5": (["no_chase", "no_overbought", "no_falling_knife", "volume_ok"], 5, 5)}
rows = []
for name, (rules, cap, cd) in V.items():
    g = apply_rules(sig, ds, rules, cap, sectors)
    s = S.simulate(g, uni, a, b, S.real_world(1_000_000.0, max_pos=30, stop_atr=4.0, sizing="invvol", cooldown_days=cd))["stats"]
    rows.append(dict(variant=name, ret=s["return"], sharpe=s["sharpe"], dd=s["max_drawdown"], trades=s["trades"])); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("runs/v4_rules.csv", index=False); print("DONE", flush=True)
