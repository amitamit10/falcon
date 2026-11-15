"""Falcon v3 under stricter real-world rules, and whether a take-profit order (a 2nd daily action) helps. Choose on 2024, report 2025."""
import sys, pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
from tradingbot.strategies import FALCON_V2
ds, uni, bench = F.load_all("data/tase125.txt")
out = Path("runs/falcon_local"); r = F.default_recipes()["cross"]
PER = {"2024 (validation)": ("2024-01-02", "2024-12-31"), "2025 (test)": ("2025-01-02", "2025-12-31")}
OLD = dict(settle_days=0, min_commission=0.0, thin_adv=0.0)
rows = []
for name, (a, b) in PER.items():
    p1, p10 = F.recipe_signals(ds, r, a, b, out)
    sig = rank_signals(p1, p10, 20, 60, 0.7)
    for label, kw in [("old real-world", OLD), ("strict real-world", {}), ("strict + take-profit 3 ATR", dict(tp_atr=3.0)), ("strict + take-profit 5 ATR", dict(tp_atr=5.0)),
                      ("strict + fill close", dict(fill="close")), ("strict + fill during day", dict(fill="ohlc4"))]:
        p = S.real_world(1_000_000.0, max_pos=20, stop_atr=3.0, **kw)
        s = S.simulate(sig, uni, a, b, p)["stats"]
        rows.append(dict(period=name, variant=label, ret=s["return"], after_tax=s["after_tax_return"], dd=s["max_drawdown"], sharpe=s["sharpe"], trades=s["trades"], fees=s["fees_paid"]))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("runs/v4_realism.csv", index=False)
