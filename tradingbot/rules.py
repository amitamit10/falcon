"""'Dumb' rule guards on top of the model's ranking: plain code, no learning. A rule can only CANCEL a buy (a held stock is never force-sold by a rule);
each looks at the stock's own numbers at the decision close, so nothing here sees the future.
  no_chase          skip a stock that jumped more than 6% on the day, or gapped up more than 4% (the move is already priced in)
  no_overbought     skip RSI(14) above 80, or more than 25% above its 50-day average
  no_falling_knife  skip a stock below a falling 150-day average
  volume_ok         skip a stock whose volume today is under 40% of normal (a thin, unreliable price)
  sector_cap        at most k new buys per sector per day"""
from __future__ import annotations

import pandas as pd

RULES = {
    "no_chase": lambda d: (d["ret_1d"].fillna(0) <= 0.06) & (d["gap"].fillna(0) <= 0.04),
    "no_overbought": lambda d: (d["rsi14"].fillna(50) <= 80) & (d["dist_ma50"].fillna(0) <= 0.25),
    "no_falling_knife": lambda d: d["trend_ok"].fillna(1) >= 1,
    "volume_ok": lambda d: d["vol_ratio"].fillna(1) >= 0.4,
}


def apply_rules(sig: pd.DataFrame, ds: pd.DataFrame, rules: list[str] = (), sector_cap: int = 0, sectors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns a copy of the signal frame with buys cancelled where a rule says no. sector_cap keeps the best-scored k buys per sector and day."""
    out = sig.copy()
    need = ["date", "ticker", "ret_1d", "gap", "rsi14", "dist_ma50", "trend_ok", "vol_ratio"]
    d = out[["date", "ticker"]].merge(ds[[c for c in need if c in ds.columns]], on=["date", "ticker"], how="left")
    for c in need[2:]:
        if c not in d: d[c] = float("nan")
    ok = pd.Series(True, index=out.index)
    for r in rules:
        ok &= RULES[r](d).to_numpy()
    out["buy"] = out["buy"] & ok.to_numpy()
    if sector_cap > 0 and sectors is not None:
        out["_sec"] = out["ticker"].map(dict(zip(sectors["ticker"], sectors["sector"]))).fillna("Unknown")
        b = out[out["buy"]].copy(); b["_r"] = b.groupby(["date", "_sec"])["score"].rank(ascending=False, method="first")
        out.loc[b[b["_r"] > sector_cap].index, "buy"] = False
        out = out.drop(columns="_sec")
    return out
