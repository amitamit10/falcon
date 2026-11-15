"""Average the rankings of several recipes into one (ensemble of ensembles); scored on 2024 only, strict real-world rules, v4's portfolio settings."""
import itertools, pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt")
a, b = "2024-01-02", "2024-12-31"; R = F.default_recipes()
SRC = {"cross": ("runs/falcon_local", (a, b)), "cross2r": ("runs/falcon_v31", ("2023-01-03", "2024-12-31")), "pro": ("runs/falcon_v31", ("2023-01-03", "2024-12-31")), "cross3": ("runs/falcon_v31", ("2023-01-03", "2024-12-31"))}
sg = {}
for k, (out, per) in SRC.items():
    p1, p10 = F.recipe_signals(ds, R[k], *per, Path(out)); sg[k] = [x[(x["date"] >= a) & (x["date"] <= b)] for x in (p1, p10)]
def blend(names):
    out = []
    for q in (0, 1):
        fr = [sg[n][q].assign(p=lambda d: d.groupby("date")["p"].rank(pct=True)).rename(columns={"p": n}) for n in names]
        m = fr[0]
        for f in fr[1:]: m = m.merge(f, on=["date", "ticker"])
        m["p"] = m[list(names)].mean(axis=1); out.append(m[["date", "ticker", "p"]])
    return out
rows = []
for names in [("cross2r",), ("cross",), ("pro",), ("cross3",), ("cross2r", "pro"), ("cross", "cross2r"), ("cross", "cross2r", "pro"), ("cross", "cross2r", "pro", "cross3")]:
    p1, p10 = blend(names) if len(names) > 1 else sg[names[0]]
    s = S.simulate(rank_signals(p1, p10, 30, 90, 0.7), uni, a, b, S.real_world(1_000_000.0, max_pos=30, stop_atr=4.0, sizing="invvol"))["stats"]
    rows.append(dict(blend="+".join(names), ret=s["return"], sharpe=s["sharpe"], dd=s["max_drawdown"], trades=s["trades"])); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("runs/v4_ensemble.csv", index=False); print("DONE", flush=True)
