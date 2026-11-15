"""Builds dataset/: the project's OWN outputs only (model scores, trade logs, equity curves, experiment tables). No raw price or financial data from third parties.
Needs the cached walk-forward predictions in runs/ (produced by `python -m tradingbot falcon ...` and scripts/v4_final.py)."""
import json, shutil
import pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals

ds, uni, bench = F.load_all("data/tase125.txt"); last = str(ds["date"].max().date())
D = Path("dataset"); (D / "signals").mkdir(parents=True, exist_ok=True); (D / "results").mkdir(exist_ok=True); (D / "universe").mkdir(exist_ok=True)
R = F.default_recipes()
CFG = {"v3": dict(recipe="cross", out="runs/falcon_local", n=20, dm=3, stop=3.0, sizing="equal", spans=[("2025-01-02", last)]),
       "v4": dict(recipe="cross2r", out="runs/falcon_v31", n=30, dm=3, stop=4.0, sizing="invvol", spans=[("2023-01-03", "2024-12-31"), ("2025-01-02", last)])}
PER = {"calendar_2025": ("2025-01-02", "2025-12-31"), "2026_so_far": ("2026-01-02", last), "past_year": ("2025-11-12", last)}
sig = {}
for k, c in CFG.items():
    parts = [F.recipe_signals(ds, R[c["recipe"]], a, b, Path(c["out"])) for a, b in c["spans"]]
    p1 = pd.concat([p[0] for p in parts]); p10 = pd.concat([p[1] for p in parts])
    sg = rank_signals(p1, p10, c["n"], c["n"] * c["dm"], 0.7).rename(columns={"score": "score_blend"})
    m = p1.rename(columns={"p": "p_next_day_up"}).merge(p10.rename(columns={"p": "p_10day_beats_median"}), on=["date", "ticker"]).merge(sg[["date", "ticker", "score_blend", "rank"]], on=["date", "ticker"])
    m["date"] = pd.to_datetime(m["date"]).dt.date; m = m.sort_values(["date", "rank"]); m.to_csv(D / "signals" / f"falcon_{k}_scores.csv.gz", index=False, float_format="%.5f")
    sig[k] = (p1, p10)
summary = {}
for name, (a, b) in PER.items():
    ix = bench[(bench.index >= a) & (bench.index <= b)]; bh = S.buy_and_hold(uni, a, b, 1.0)
    eq = pd.DataFrame({"ta125_index": ix / ix.iloc[0], "equal_weight_buy_and_hold": bh / bh.iloc[0]})
    summary[name] = {"ta125_index_return": float(ix.iloc[-1] / ix.iloc[0] - 1), "equal_weight_buy_and_hold_return": float(bh.iloc[-1] / bh.iloc[0] - 1)}
    for k, c in CFG.items():
        p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in sig[k]]
        r = S.simulate(rank_signals(p1, p10, c["n"], c["n"] * c["dm"], 0.7), uni, a, b, S.real_world(1_000_000.0, max_pos=c["n"], stop_atr=c["stop"], sizing=c["sizing"]))
        eq[f"falcon_{k}_equity_ils"] = r["equity"]; r["trades"].to_csv(D / "results" / f"{name}_falcon_{k}_trades.csv", index=False, float_format="%.5f")
        summary[name][f"falcon_{k}"] = {x: float(y) for x, y in r["stats"].items()}
    eq.index.name = "date"; eq.to_csv(D / "results" / f"{name}_equity.csv", float_format="%.5f")
(D / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
for f in ("tase125.txt",): shutil.copy(f"data/{f}", D / "universe" / f)
for f in ("sectors.csv", "dual_listed.csv"): shutil.copy(f"data/{f}", D / "universe" / f)
for src, dst in [("runs/v4_realism.csv", "realism_checks_v3.csv"), ("runs/v4_search_cross2r.csv", "portfolio_search_2024_cross2r.csv"), ("runs/v4_ideas.csv", "extra_ideas_2024.csv"),
                 ("runs/v4_rules.csv", "rule_guards_2024.csv"), ("runs/v4_ensemble.csv", "recipe_ensembles_2024.csv"), ("docs/falcon_v3_select_recipes.csv", "recipe_selection_2024.csv")]:
    if Path(src).exists(): shutil.copy(src, D / "results" / f"experiments_{dst}")
print("DONE", flush=True)
