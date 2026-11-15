"""Falcon v4 vs v3: ONE run on the test periods, settings fixed beforehand (chosen on 2023-24 / 2024 only). Strict real-world rules."""
import json, pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt")
last = str(ds["date"].max().date())
PER = {"Calendar 2025": ("2025-01-02", "2025-12-31"), "2026 so far": ("2026-01-02", last), "Past year": ("2025-11-12", last)}
V3 = dict(recipe="cross", out="runs/falcon_local", n=20, dm=3, stop=3.0, w10=0.7, sizing="equal")
V4 = dict(recipe="cross2r", out="runs/falcon_v31", n=30, dm=3, stop=4.0, w10=0.7, sizing="invvol")
res = {}
SPAN = ("2025-01-02", last); cache = {}
def signals(c):
    if c["recipe"] not in cache:                                      # walk-forward over the whole test span once, then sliced per period
        cache[c["recipe"]] = F.recipe_signals(ds, F.default_recipes()[c["recipe"]], *SPAN, Path(c["out"]))
    return cache[c["recipe"]]
for name, (a, b) in PER.items():
    res[name] = {}
    for label, c in (("Falcon v3", V3), ("Falcon v4", V4)):
        p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in signals(c)]
        sig = rank_signals(p1, p10, c["n"], c["n"] * c["dm"], c["w10"])
        for scen, kw in (("Real-world (strict)", {}), ("Real-world, order during the day", dict(fill="ohlc4"))):
            st = S.simulate(sig, uni, a, b, S.real_world(1_000_000.0, max_pos=c["n"], stop_atr=c["stop"], sizing=c["sizing"], **kw))["stats"]
            res[name][f"{label} | {scen}"] = {k: st[k] for k in ("return", "after_tax_return", "max_drawdown", "sharpe", "trades", "fees_paid")}
            print(name, label, scen, {k: round(v, 3) for k, v in res[name][f"{label} | {scen}"].items()}, flush=True)
    bh = S.buy_and_hold(uni, a, b, 1.0); ix = bench[(bench.index >= a) & (bench.index <= b)]
    res[name]["benchmarks"] = dict(buy_and_hold=float(bh.iloc[-1] / bh.iloc[0] - 1), index=float(ix.iloc[-1] / ix.iloc[0] - 1))
    print(name, res[name]["benchmarks"], flush=True)
json.dump(res, open("runs/v4_final.json", "w"), indent=1); print("DONE", flush=True)
