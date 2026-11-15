"""Chart: Falcon v3 and v4 (strict real-world, TA-125 only, 1,000,000 ILS) vs the index and buy&hold, with the professional-fund levels. Uses the cached predictions."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt"); last = str(ds["date"].max().date())
PER = {"Calendar 2025": ("2025-01-02", "2025-12-31"), "2026 so far": ("2026-01-02", last), "Past year": ("2025-11-12", last)}
CFG = {"Falcon v3": dict(recipe="cross", out="runs/falcon_local", n=20, dm=3, stop=3.0, sizing="equal", color="#2a7ab0"),
       "Falcon v4": dict(recipe="cross2r", out="runs/falcon_v31", n=30, dm=3, stop=4.0, sizing="invvol", color="#d4791f")}
sg = {k: F.recipe_signals(ds, F.default_recipes()[c["recipe"]], "2025-01-02", last, Path(c["out"])) for k, c in CFG.items()}
fig, axs = plt.subplots(1, 3, figsize=(18, 5.6)); M = 1_000_000.0
for ax, (name, (a, b)) in zip(axs, PER.items()):
    for k, c in CFG.items():
        p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in sg[k]]
        r = S.simulate(rank_signals(p1, p10, c["n"], c["n"] * c["dm"], 0.7), uni, a, b, S.real_world(M, max_pos=c["n"], stop_atr=c["stop"], sizing=c["sizing"]))
        e = r["equity"]; ax.plot(e.index, e / M - 1, color=c["color"], lw=2, label=f"{k}: {r['stats']['return']:+.0%} ({r['stats']['after_tax_return']:+.0%} after tax)")
    ix = bench[(bench.index >= a) & (bench.index <= b)]; ax.plot(ix.index, ix / ix.iloc[0] - 1, color="#555", lw=1.3, label=f"TA-125 index: {ix.iloc[-1] / ix.iloc[0] - 1:+.0%}")
    bh = S.buy_and_hold(uni, a, b, 1.0); ax.plot(bh.index, bh / bh.iloc[0] - 1, color="#999", lw=1, ls="--", label=f"Equal-weight buy&hold: {bh.iloc[-1] / bh.iloc[0] - 1:+.0%}")
    for lab, v in F.PRO_BENCHMARKS.get(name, []):
        ax.axhline(v, color="#8e2c8e", lw=1, ls=(0, (1, 2))); ax.annotate(f"{lab}: {v:+.0%}", xy=(0.995, v), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=7, color="#8e2c8e")
    ax.axhline(0, color="#999", lw=.6); ax.grid(alpha=.25); ax.set_title(name, fontsize=11); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Falcon v3 vs v4 - paper trading, strict real-world rules (T+1, commissions, spreads, tax not deducted in lines), TA-125 stocks only", fontsize=11)
fig.text(0.01, 0.005, "Simulation on past data, not real money. Professional-fund levels are 2025 figures from press reports, gross, long/short, not like-for-like.", fontsize=8, color="#666")
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig("docs/falcon_v3_v4.png", dpi=150); print("CHART DONE")
