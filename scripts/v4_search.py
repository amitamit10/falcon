"""Falcon v4 portfolio search: wider grid than v3 (its best cell sat on the grid edge), strict real-world rules, chosen on 2024 only (Sharpe, drawdown <= 25%)."""
import itertools, json, pandas as pd
from pathlib import Path
from tradingbot import falcon as F, simulate as S
from tradingbot.selective import rank_signals
ds, uni, bench = F.load_all("data/tase125.txt")
import sys
rec = sys.argv[1] if len(sys.argv) > 1 else "cross"; out = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/falcon_local")
r = F.default_recipes()[rec]
va, vb = (sys.argv[3].split(",") if len(sys.argv) > 3 else ("2024-01-02", "2024-12-31"))
a, b = "2024-01-02", "2024-12-31"                                              # the portfolio is chosen on 2024 only
p1, p10 = F.recipe_signals(ds, r, va, vb, out)
p1, p10 = [x[(x["date"] >= a) & (x["date"] <= b)] for x in (p1, p10)]
rows = []
for n, dm, stop, w10, sizing in itertools.product((15, 20, 30, 40), (3, 4, 6), (3.0, 4.0), (0.5, 0.7, 0.85), ("equal", "invvol")):
    sig = rank_signals(p1, p10, n, n * dm, w10)
    s = S.simulate(sig, uni, a, b, S.real_world(1_000_000.0, max_pos=n, stop_atr=stop, sizing=sizing))["stats"]
    rows.append(dict(n=n, dm=dm, stop=stop, w10=w10, sizing=sizing, ret=s["return"], sharpe=s["sharpe"], dd=s["max_drawdown"], trades=s["trades"]))
    print(rows[-1], flush=True)
g = pd.DataFrame(rows); g.to_csv(f"runs/v4_search_{rec}.csv", index=False)
ok = g[g["dd"] >= -F.MAX_DD_FOR_SELECTION].sort_values("sharpe", ascending=False)
print("BEST", ok.head(8).to_string(), flush=True)
json.dump(ok.iloc[0].to_dict(), open(f"runs/v4_choice_{rec}.json", "w"))
