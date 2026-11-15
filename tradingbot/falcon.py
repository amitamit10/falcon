"""Falcon v3 research pipeline: one entry point that runs the same way on a laptop and on Google Colab, and survives being interrupted.

    python -m tradingbot falcon select --out runs/falcon     # choose the v3 signal recipe and portfolio settings on the VALIDATION year only
    python -m tradingbot falcon final  --out runs/falcon     # run v1, v2, v3 once on the two test periods, ideal and real-world, write report + chart
    python -m tradingbot falcon all    --out runs/falcon     # both

Protocol (nothing here looks at a test period before the choice is made):
  1. `select`: every candidate recipe makes walk-forward predictions for the validation year (2024). Each model is trained only on labels that were already known on
     the day it predicts, and is retrained every `every` trading days. The recipe with the best rank-IC (70% the 10-day question, 30% next-day) wins, then the portfolio
     settings (positions, hold rank, stop) are searched for that recipe on the same year under real-world costs.
  2. `final`: Falcon v1 (baseline), v2 (portfolio settings chosen earlier) and v3 (this selection) run once on calendar 2025 and the past year, under three scenarios:
     Ideal (0.1% a trade, next open), Real-world (spread, market impact, liquidity cap, whole shares, interest on cash, 25% tax, in shekels) and
     Real-world, order during the day (same morning, filled at the day's average price) and Real-world, one day late (a full extra day lost: a stress test, not an expected delay).

Every model's predictions are written to disk block by block, so a Colab disconnect only costs the block in progress: run the same command again and it resumes.
All of it is a simulation with paper money; see docs/SIMULATION_ONE_YEAR.md for what the numbers can and cannot tell you."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import simulate as S
from .forward import evaluate
from .selective import rank_signals
from .strategies import FALCON_V1, FALCON_V2, Strategy

BENCH = "^TA125.TA"
VAL = ("2024-01-02", "2024-12-31")
TESTS = {"Calendar 2025": ("2025-01-02", "2025-12-31"), "Past year": ("2025-11-12", "2026-11-11")}
SCENARIOS = ("Ideal", "Real-world", "Real-world, order during the day", "Real-world, one day late (stress test)")
GRID = dict(n=(10, 15, 20), dm=(2, 3), stop=(2.0, 3.0))
W10 = 0.7                                  # weight of the 10-day question in the daily ranking (the next-day question gets the rest)
MAX_DD_FOR_SELECTION = 0.25
MIN_STOCKS = 30                            # rank IC needs at least this many stocks on a day (the real universe has 117)
CLASSIFIER_TARGETS = {"label_up1", "label_xs"}
Q = {"1d": dict(target="label_up1", date_col="label_date1", y="label_up1", ret="ret_fwd1", horizon=1),
     "10d": dict(target="label_xs", date_col="label_date", y="label_xs", ret="excess_ret", horizon=10)}


# Published results of professional managers, for scale. NOT like-for-like: mutual hedge funds hold 0-100% shares and may use leverage; returns are gross (after management fees,
# before the roughly 20% success fee). Source: Globes, "Mutual hedge funds approach NIS 4b", 20 Jan 2026 (calendar 2025). TA-125 +50% in 2025 per ynet, 29 Dec 2025.
PRO_BENCHMARKS = {"Calendar 2025": [("Harel Multi-Strategy (top mutual hedge fund)", 0.78), ("Meitav Long Short (mutual hedge fund)", 0.68),
                                    ("Average of 46 Israeli mutual hedge funds", 0.25)]}


def log(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------------- models and recipes
@dataclass(frozen=True)
class Member:
    """One model of an ensemble: question ('1d' next-day up, '10d' beats the median stock over 10 days), algorithm, feature set, and how it is retrained."""
    q: str
    algo: str
    fs: str
    target: str = ""            # empty = the question's classifier target; 'rank_q' (lambdarank) and 'rank_target' (rank regression) are 10-day only
    every: int = 21             # retrain every this many trading days
    window: int | None = 6      # training window in years (None = all history)

    @property
    def tgt(self) -> str:
        return self.target or Q[self.q]["target"]

    @property
    def date_col(self) -> str:
        return Q[self.q]["date_col"] if self.tgt == Q[self.q]["target"] else "label_date"

    @property
    def is_classifier(self) -> bool:
        return self.tgt in CLASSIFIER_TARGETS

    @property
    def key(self) -> str:
        return f"{self.q}-{self.algo}-{self.fs}-{self.tgt}-e{self.every}-w{self.window or 'all'}"


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    members: tuple

    def of(self, q: str) -> list[Member]:
        return [m for m in self.members if m.q == q]


def _has(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def default_recipes() -> dict[str, Recipe]:
    """The candidates. 'base' is exactly the signal engine behind Falcon v1 and v2."""
    M = Member
    base1 = [M("1d", "lgbm", "xs"), M("1d", "lgbm", "xsx"), M("1d", "lgbm", "xsxa")]
    base10 = [M("10d", "hgb", "xsx"), M("10d", "lgbm", "xsx"), M("10d", "lgbm", "xsxa")]
    rank10 = [M("10d", "lgbm_rank", "xsx", target="rank_q"), M("10d", "lgbm_reg", "xsx", target="rank_target")]
    cross1 = [M("1d", "lgbm", "xs"), M("1d", "lgbm", "xsd"), M("1d", "lgbm", "xsda")]
    cross10 = [M("10d", "hgb", "xsd"), M("10d", "lgbm", "xsd"), M("10d", "lgbm", "xsda")]
    cross2_1 = [M("1d", "lgbm", "xs"), M("1d", "lgbm", "xss"), M("1d", "lgbm", "xssa")]
    cross2_10 = [M("10d", "hgb", "xss"), M("10d", "lgbm", "xss"), M("10d", "lgbm", "xssa")]
    rank10s = [M("10d", "lgbm_rank", "xss", target="rank_q"), M("10d", "lgbm_reg", "xss", target="rank_target")]
    cross3_1 = [M("1d", "lgbm", "xs"), M("1d", "lgbm", "xsf"), M("1d", "lgbm", "xsfa")]
    cross3_10 = [M("10d", "hgb", "xsf"), M("10d", "lgbm", "xsf"), M("10d", "lgbm", "xsfa")]
    pro_1 = [M("1d", "lgbm", "xs"), M("1d", "lgbm", "xsp"), M("1d", "lgbm", "xspa")]
    pro_10 = [M("10d", "hgb", "xsp"), M("10d", "lgbm", "xsp"), M("10d", "lgbm", "xspa"), M("10d", "lgbm_rank", "xsp", target="rank_q"), M("10d", "lgbm_reg", "xsp", target="rank_target")]
    tgt_10 = [M("10d", "lgbm_reg", "xss", target="rk_gauss"), M("10d", "lgbm_reg", "xssa", target="rk_gauss"), M("10d", "lgbm_reg", "xss", target="sn_gauss"), M("10d", "lgbm_reg", "xssa", target="sn_gauss")]
    big1, big10 = [M("1d", "xgb_big", "xsx")], [M("10d", "xgb_big", "xsx")]
    R: dict[str, Recipe] = {}

    def add(name, desc, members):
        R[name] = Recipe(name, desc, tuple(members))

    add("base", "Falcon v1/v2 engine: 3 models per question, retrained every 21 days, 6-year window", base1 + base10)
    add("fast10", "base, retrained every 10 days instead of 21", [replace(m, every=10) for m in base1 + base10])
    add("ens5", "base plus a lambdarank and a rank-regression model on the 10-day question", base1 + base10 + rank10)
    add("cross", "base with the cross-listing inputs (US prices of dual-listed companies, EIS, semiconductors)", cross1 + cross10)
    add("cross2", "cross plus US sector ETFs mapped to each stock's sector and gold / dollar / credit / long-bond / EUR-ILS flow", cross2_1 + cross2_10)
    add("cross2r", "cross2 plus a lambdarank and a rank-regression model on the 10-day question", cross2_1 + cross2_10 + rank10s)
    add("tgt", "cross2r plus regressions on a rank-Gauss target and on a sector-neutral rank-Gauss target (target engineering)", cross2_1 + cross2_10 + rank10s + tgt_10)
    add("pro", "cross2r plus the setups chart traders screen for: volatility contraction, volume dry-up and breakout, relative-strength line, 52-week-low distance, trend template", pro_1 + pro_10)
    add("cross3", "cross2 plus quarterly company reports (growth, margins, cash flow, leverage, valuation re-scaled to today's price; usable 60-90 days after each quarter end)", cross3_1 + cross3_10)
    add("longwin", "base trained on all history instead of the last 6 years", [replace(m, window=None) for m in base1 + base10])
    if _has("xgboost"):
        add("big", "base plus a large XGBoost (800 trees, depth 6; on the GPU when there is one) per question", base1 + big1 + base10 + big10)
        add("full", "cross inputs + rank models + large XGBoost, retrained every 10 days", [replace(m, every=10) for m in cross1 + big1 + cross10 + rank10 + big10])
    return R


def combine(preds: list[pd.DataFrame], members: list[Member]) -> pd.DataFrame:
    """Average the members of one question. Classifiers are averaged as probabilities; as soon as a rank / regression model is in the mix every member is first turned
    into its within-day percentile rank (their raw scores live on different scales). Only stock-days every member predicted are kept."""
    use_rank = any(not m.is_classifier for m in members)
    cols = []
    for pr in preds:
        s = pr.drop_duplicates(["date", "ticker"]).set_index(["date", "ticker"])["p"]
        cols.append(s.groupby(level=0).rank(pct=True) if use_rank else s)
    p = pd.concat(cols, axis=1, join="inner").mean(axis=1)
    return p.rename("p").reset_index()


def member_predictions(ds: pd.DataFrame, m: Member, start: str, end: str, out_dir: Path) -> pd.DataFrame:
    """Walk-forward predictions of one model for every stock-day in [start, end] (labels are not needed, so the newest days are included). A block of `every` days
    is predicted by a model trained on the labels known at the block's first day. Blocks are cached on disk and skipped when present."""
    dates = sorted(ds.loc[(ds["date"] >= start) & (ds["date"] <= end), "date"].unique())
    folder = Path(out_dir) / "members" / m.key / f"{start}_{end}"
    folder.mkdir(parents=True, exist_ok=True)
    nblocks, parts, t0, fitted = (len(dates) + m.every - 1) // m.every, [], time.time(), 0
    for bi, i in enumerate(range(0, len(dates), m.every)):
        f = folder / f"block_{bi:04d}.csv.gz"
        if f.exists():
            parts.append(pd.read_csv(f, parse_dates=["date"]))
            continue
        block = dates[i:i + m.every]
        rows = ds[ds["date"].isin(block)].dropna(subset=["rsi14", "dist_ma50"])
        from .model import predict, train
        try:
            model = train(ds, pd.Timestamp(block[0]), m.window, target=m.tgt, date_col=m.date_col, feature_set=m.fs, algo=m.algo)
        except ValueError:
            continue                                               # not enough labelled history before this block
        out = pd.DataFrame({"date": rows["date"].to_numpy(), "ticker": rows["ticker"].to_numpy(), "p": np.asarray(predict(model, rows), dtype=float)})
        tmp = f.with_name(f.name + ".tmp")
        out.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, f)                                         # atomic: an interrupted write never leaves a half block
        parts.append(out)
        fitted += 1
        eta = (time.time() - t0) / fitted * (nblocks - bi - 1)
        log(f"    {m.key}: block {bi + 1}/{nblocks} done, ~{eta / 60:.1f} min left for this model")
    if not parts:
        raise ValueError(f"no predictions for {m.key} between {start} and {end}")
    return pd.concat(parts, ignore_index=True)


def recipe_signals(ds: pd.DataFrame, r: Recipe, start: str, end: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(p1, p10): the ensemble predictions for the two questions. Frames with date, ticker, p."""
    out = []
    for q in ("1d", "10d"):
        ms = r.of(q)
        out.append(combine([member_predictions(ds, m, start, end, out_dir) for m in ms], ms))
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------------------------- evaluation
def signal_quality(ds: pd.DataFrame, p1: pd.DataFrame, p10: pd.DataFrame, start: str, end: str) -> dict:
    """AUC and rank IC of each question against what really happened, on [start, end] (needs known outcomes, so it is used on the validation year)."""
    out = {}
    for q, p in (("1d", p1), ("10d", p10)):
        spec = Q[q]
        lab = ds[["date", "ticker", spec["y"], spec["ret"]]].rename(columns={spec["y"]: "y", spec["ret"]: "ret"})
        m = p.merge(lab, on=["date", "ticker"]).dropna(subset=["y", "ret", "p"])
        m = m[(m["date"] >= start) & (m["date"] <= end)]
        e = evaluate(m, horizon=spec["horizon"], min_stocks=MIN_STOCKS)
        out[q] = dict(auc=float(e["auc"]), ic=float(e["rank_IC"]), ic_t=float(e["IC_t"]), n=int(e["n"]))
    out["score"] = W10 * out["10d"]["ic"] + (1 - W10) * out["1d"]["ic"]
    return out


def make_params(st: Strategy, scenario: str) -> S.Params:
    """Simulator settings for a strategy under one of the three scenarios."""
    if scenario == "Ideal":
        return st.params()
    kw = dict(max_pos=st.top_n, stop_atr=st.stop_atr, trail_atr=st.trail_atr, sizing=st.sizing)
    if scenario == "Real-world":
        return S.real_world(1_000_000.0, **kw)
    if scenario == "Real-world, order during the day":
        return S.real_world(1_000_000.0, fill="ohlc4", **kw)                       # the same morning, but the order is placed during the session: filled at the day's average price
    if scenario == "Real-world, one day late (stress test)":
        return S.real_world(1_000_000.0, delay_days=1, fill="ohlc4", **kw)        # a full extra day lost: a pessimistic worst case, not an expected delay
    raise ValueError(f"unknown scenario {scenario}")


def run_sim(p1: pd.DataFrame, p10: pd.DataFrame, st: Strategy, uni: dict, start: str, end: str, scenario: str) -> dict:
    sig = rank_signals(p1, p10, st.top_n, st.drop_rank, st.w10)
    return S.simulate(sig, uni, start, end, make_params(st, scenario))


def search_portfolio(p1: pd.DataFrame, p10: pd.DataFrame, uni: dict, start: str, end: str, scenario: str = "Real-world") -> tuple[Strategy, pd.DataFrame]:
    """Best (positions, hold-rank multiple, stop) on the validation year: highest Sharpe among settings whose drawdown stays within the limit."""
    rows = []
    for n in GRID["n"]:
        for dm in GRID["dm"]:
            for stop in GRID["stop"]:
                st = Strategy("candidate", "", top_n=n, drop_mult=dm, w10=W10, stop_atr=stop)
                s = run_sim(p1, p10, st, uni, start, end, scenario)["stats"]
                rows.append(dict(n=n, dm=dm, stop=stop, ret=s["return"], sharpe=s["sharpe"], dd=s["max_drawdown"], trades=s["trades"]))
    g = pd.DataFrame(rows)
    ok = g[g["dd"] >= -MAX_DD_FOR_SELECTION]
    best = (ok if len(ok) else g).sort_values("sharpe", ascending=False).iloc[0]
    return Strategy("Falcon v3", "", top_n=int(best["n"]), drop_mult=int(best["dm"]), w10=W10, stop_atr=float(best["stop"])), g.sort_values("sharpe", ascending=False)


# ------------------------------------------------------------------------------------------------------------------- stages
def run_select(ds: pd.DataFrame, uni: dict, out_dir: Path, recipes: dict[str, Recipe], val: tuple = VAL) -> dict:
    rows, sigs = [], {}
    for name, r in recipes.items():
        t0 = time.time()
        log(f"[select] recipe '{name}': {r.description}")
        p1, p10 = recipe_signals(ds, r, val[0], val[1], out_dir)
        q = signal_quality(ds, p1, p10, *val); sigs[name] = (p1, p10)
        rows.append(dict(recipe=name, auc1=q["1d"]["auc"], ic1=q["1d"]["ic"], ic1_t=q["1d"]["ic_t"], auc10=q["10d"]["auc"], ic10=q["10d"]["ic"], ic10_t=q["10d"]["ic_t"],
                         score=q["score"], minutes=(time.time() - t0) / 60, description=r.description))
        log(f"[select]   AUC next-day {q['1d']['auc']:.4f}, 10-day {q['10d']['auc']:.4f}; rank IC {q['1d']['ic']:.4f} / {q['10d']['ic']:.4f}; score {q['score']:.4f}")
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if table["score"].isna().all():
        raise RuntimeError("every recipe scored NaN: too few stocks per day for a rank IC, or no overlap between predictions and outcomes")
    best = str(table.iloc[0]["recipe"])
    st, grid = search_portfolio(*sigs[best], uni, *val)
    base_ref = None
    if "base" in sigs:
        base_st, _ = search_portfolio(*sigs["base"], uni, *val)
        b = run_sim(*sigs["base"], base_st, uni, *val, "Real-world")["stats"]
        base_ref = dict(strategy=f"{base_st.top_n}/{base_st.drop_rank}/{base_st.stop_atr}", ret=b["return"], sharpe=b["sharpe"])
    chosen = dict(recipe=best, n=st.top_n, dm=st.drop_mult, stop=st.stop_atr, w10=W10, val=list(val), table=table.to_dict("records"), grid=grid.head(10).to_dict("records"),
                  base_reference=base_ref, validation_real_world=run_sim(*sigs[best], st, uni, *val, "Real-world")["stats"])
    (Path(out_dir) / "select.json").write_text(json.dumps(chosen, indent=1, default=float))
    table.to_csv(Path(out_dir) / "select_recipes.csv", index=False)
    log(f"[select] chosen recipe: {best}; portfolio: {st.top_n} positions, hold inside the top {st.drop_rank}, {st.stop_atr}-ATR stop")
    return chosen


def pit_members(allu: dict, top: int = 125) -> pd.MultiIndex:
    """(date, ticker) pairs of the `top` most liquid stocks by trailing 60-day traded value on that date, among every stock supplied: a point-in-time universe."""
    turn = pd.DataFrame({tk: (df["close"] * df["volume"]).rolling(60).median() for tk, df in allu.items() if len(df) > 100})
    m = (turn.rank(axis=1, ascending=False) <= top).stack()
    return m[m].index.set_names(["date", "ticker"])


def _filter(p: pd.DataFrame, members: pd.MultiIndex) -> pd.DataFrame:
    return p[pd.MultiIndex.from_frame(p[["date", "ticker"]]).isin(members)]


def run_final(ds: pd.DataFrame, uni: dict, bench: pd.Series, out_dir: Path, chosen: dict, recipes: dict[str, Recipe], tests: dict = TESTS, allu: dict | None = None) -> dict:
    start, end = min(t[0] for t in tests.values()), max(t[1] for t in tests.values())
    base, v3r = recipes["base"], recipes[chosen["recipe"]]
    log(f"[final] signals for the base engine (v1, v2) and for '{v3r.name}' (v3), {start} to {end}")
    sig_base = recipe_signals(ds, base, start, end, out_dir)
    sig_v3 = sig_base if v3r.name == "base" else recipe_signals(ds, v3r, start, end, out_dir)
    v3 = Strategy("Falcon v3", f"signals '{v3r.name}', {chosen['n']} positions, hold inside top {chosen['n'] * chosen['dm']}, {chosen['stop']}-ATR stop",
                  top_n=int(chosen["n"]), drop_mult=int(chosen["dm"]), w10=float(chosen["w10"]), stop_atr=float(chosen["stop"]))
    runs = [(FALCON_V1, sig_base), (FALCON_V2, sig_base), (v3, sig_v3)]
    members = pit_members(allu) if allu else None
    results, curves, trades = {}, {}, {}
    for period, (a, b) in tests.items():
        px = bench[(bench.index >= a) & (bench.index <= b)]
        bh = S.buy_and_hold(uni, a, b)
        results[period] = {"benchmarks": {"buy_and_hold": float(bh.iloc[-1] / 100_000 - 1), "index": float(px.iloc[-1] / px.iloc[0] - 1)}}
        curves[period] = {"bh": bh, "index": 100_000 * px / px.iloc[0]}
        for scen in SCENARIOS:
            res = results[period].setdefault(scen, {})
            for st, (p1, p10) in runs:
                r = run_sim(p1, p10, st, uni, a, b, scen)
                res[st.name] = {k: float(v) for k, v in r["stats"].items()}
                curves[period][(scen, st.name)] = r["equity"] / r["equity"].iloc[0]
                if scen == "Real-world":
                    trades[(period, st.name)] = r["trades"]
        if members is not None:
            pit = results[period].setdefault("Real-world, point-in-time universe", {})
            for st, (p1, p10) in runs:
                pit[st.name] = {k: float(v) for k, v in run_sim(_filter(p1, members), _filter(p10, members), st, uni, a, b, "Real-world")["stats"].items()}
        log(f"[final] {period}: " + ", ".join(f"{st.name} {results[period]['Real-world'][st.name]['return']:+.0%}" for st, _ in runs) + " (real-world)")
    out = dict(chosen=chosen, v3=v3.description, results=results)
    (Path(out_dir) / "final.json").write_text(json.dumps(out, indent=1, default=float))
    for (period, name), t in trades.items():
        t.to_csv(Path(out_dir) / f"trades_{period.replace(' ', '_')}_{name.replace(' ', '_')}.csv", index=False)
    chart(curves, results, Path(out_dir) / "falcon_v1_v2_v3.png")
    write_report(out, Path(out_dir) / "report.md")
    return out


# ------------------------------------------------------------------------------------------------------------------- outputs
def chart(curves: dict, results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    periods = list(curves)
    fig, axes = plt.subplots(2, len(periods), figsize=(6.2 * len(periods), 9.2), squeeze=False)
    col = {"Falcon v1": "#2b7bba", "Falcon v2": "#0f2a43", "Falcon v3": "#1b7f4b"}
    for j, period in enumerate(periods):
        for i, scen in enumerate(("Ideal", "Real-world")):
            ax = axes[i][j]
            for name, c in col.items():
                eq = curves[period].get((scen, name))
                if eq is not None:
                    ax.plot(eq.index, 100_000 * eq, color=c, lw=2.3 if name == "Falcon v3" else 1.6, label=f"{name}: {results[period][scen][name]['return']:+.0%}")
            bh, ix = curves[period]["bh"], curves[period]["index"]
            ax.plot(bh.index, bh, color="#b26a00", lw=1.2, label=f"Buy everything, hold: {results[period]['benchmarks']['buy_and_hold']:+.0%}")
            ax.plot(ix.index, ix, color="#5f6b76", lw=1.2, ls="--", label=f"TA-125 index: {results[period]['benchmarks']['index']:+.0%}")
            for k, (label, r) in enumerate(PRO_BENCHMARKS.get(period, [])):                          # professional managers: a level line for the whole year
                ax.axhline(100_000 * (1 + r), color="#8e2c8e", lw=1.0, ls=(0, (1, 2)), alpha=.9)
                ax.annotate(f"{label}: {r:+.0%}", xy=(0.995, 100_000 * (1 + r)), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=7, color="#8e2c8e", bbox=dict(fc="white", ec="none", alpha=.85, pad=.8))
            ax.axhline(100_000, color="#999", lw=.6); ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.tick_params(labelsize=8)
            ax.set_title(f"{period} - {scen}", fontsize=10.5, color="#0f2a43")
            if j == 0:
                ax.set_ylabel("Portfolio value (start 100,000)")
    key = ("Falcon = our trading model: every day it ranks the 117 TA-125 stocks, buys the best, holds while they stay near the top, sells the rest or on a stop-loss.\n"
           "v1 baseline: 20 stocks, sell below rank 40, 2x stop.  v2: 10 stocks, sell below rank 30, 3x stop.  v3: this run's choice (settings chosen on 2024 only).\n"
           "Ideal: 0.1% a trade, next-open fills.  Real-world: spread, market impact, liquidity cap, whole shares, interest on cash, in shekels.  Curves are before tax.\n"
           "Purple dotted lines (2025 only): published results of Israeli professional mutual hedge funds (Globes, Jan 2026), gross of the ~20% success fee.\n"
           "They hold 0-100% shares and may use leverage, so this is a yardstick, not a like-for-like comparison.\n"
           "Paper money. Not a promise: flattered by using today's largest stocks, and by a few big winners.")
    fig.text(0.01, 0.005, key, fontsize=8, va="bottom", ha="left", color="#22303c")
    fig.tight_layout(rect=(0, 0.10, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


def write_report(out: dict, path: Path) -> None:
    ch, res = out["chosen"], out["results"]
    L = ["# Falcon v3 run", "", f"Chosen on the validation year {ch['val'][0]} to {ch['val'][1]} only: recipe **{ch['recipe']}**, {ch['n']} positions, hold inside the top {ch['n'] * ch['dm']}, "
         f"{ch['stop']}-ATR stop.", "", "## Signal recipes on the validation year (higher score is better; score = 0.7 x 10-day rank IC + 0.3 x next-day rank IC)",
         "", "| Recipe | AUC next day | AUC 10 days | Rank IC next day | Rank IC 10 days | Score | Minutes |", "|---|---|---|---|---|---|---|"]
    for r in ch["table"]:
        L.append(f"| {r['recipe']} | {r['auc1']:.4f} | {r['auc10']:.4f} | {r['ic1']:.4f} | {r['ic10']:.4f} | {r['score']:.4f} | {r['minutes']:.0f} |")
    if ch.get("base_reference"):
        b = ch["base_reference"]
        L += ["", f"Validation year, real-world, best portfolio for the base engine ({b['strategy']}): {b['ret']:+.0%}, Sharpe {b['sharpe']:.2f}. "
              f"Chosen v3: {ch['validation_real_world']['return']:+.0%}, Sharpe {ch['validation_real_world']['sharpe']:.2f}."]
    L += ["", "## Professional managers, for scale (calendar 2025)", "", "Published results (Globes, 20 Jan 2026): Israeli *mutual hedge funds* hold 0-100% shares and may use leverage; returns are gross (after management fees, "
          "before the ~20% success fee). They are a yardstick, not a like-for-like comparison with a long-only stock strategy. TA-125 index in 2025: +50% (ynet, 29 Dec 2025).", "",
          "| Fund | 2025 return |", "|---|---|"] + [f"| {n} | {r:+.0%} |" for n, r in PRO_BENCHMARKS["Calendar 2025"]]
    L += ["", "No published 2025 return table for plain actively managed equity funds could be retrieved (the main fund-data site refused automated access); the Calcalist half-year review "
          "said most large equity funds beat the index in the first half of 2025, without figures."]
    for period, d in res.items():
        L += ["", f"## {period}", "", f"Benchmarks: buy everything and hold {d['benchmarks']['buy_and_hold']:+.1%}, TA-125 index {d['benchmarks']['index']:+.1%}.", ""]
        for scen in [k for k in d if k != "benchmarks"]:
            L += [f"### {scen}", "", "| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |", "|---|---|---|---|---|---|---|---|"]
            for name, s in d[scen].items():
                L.append(f"| {name} | {s['return']:+.1%} | {s['after_tax_return']:+.1%} | {s['max_drawdown']:.0%} | {s['sharpe']:.2f} | {int(s['trades'])} | {s['win_rate']:.0%} | {s['fees_paid']:,.0f} |")
            L.append("")
    L += ["## What the scenarios mean (the timeline of one trade)",
          "- The model runs in the morning, before the TASE opens (05:00 UTC), once the previous US session is final: the data of day t, including the US close, is known on the morning of day t+1.",
          "- **Ideal**: the order fills at that day's opening price; only a flat 0.1% cost per trade.",
          "- **Real-world**: the same timing, as an automated order in the opening auction, but with spread, market impact, a liquidity cap, whole shares, interest on idle cash and tax. This is the intended way of running it.",
          "- **Real-world, order during the day**: the order is placed some time after the open (a person reading the report, or a slower connection): filled at that day's average price (mean of open, high, low and close). A reasonable model of a delay of a few hours.",
          "- **Real-world, one day late (stress test)**: a whole extra day is lost (the order fills the day after, at its average price). This is a pessimistic worst case, not a delay to expect.",
          "- Daily bars cannot model delays of minutes exactly; the only intraday history available (60 days of 5-minute bars) is too short to calibrate one.", ""]
    L += ["## How to read this", "- Everything is paper money. Prices are TASE agorot; real-world runs are in shekels with 1,000,000 to start.",
          "- 'After tax' applies 25% to each calendar year's net realised profit in the real-world scenarios; the Ideal scenario has no tax, so there it equals the return.",
          "- The settings were chosen on the validation year only; the test periods were run once. Earlier feature and model choices were made with 2025-26 in view, so those periods are not perfectly untouched.",
          "- Survivorship: the universe is today's largest stocks. The point-in-time section (when present) restricts each day to the stocks that were really among the 125 most liquid then."]
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------------------------- data and CLI
def load_all(tickers_path: str, refresh: bool = False, universe_all: bool = False) -> tuple[pd.DataFrame, dict, pd.Series]:
    """Prices for the universe, the feature dataset (float32 to halve the memory), and the TA-125 index close."""
    from .data import get_prices, get_universe, load_tickers
    from .features_more import augment
    from .model import build_dataset
    uni = get_universe(load_tickers(tickers_path), refresh=refresh)
    bench = get_prices(BENCH, refresh=refresh)["close"]
    ds = augment(build_dataset(uni, bench, 10), uni)
    for c in ds.columns[ds.dtypes == "float64"]:
        ds[c] = ds[c].astype("float32")
    log(f"[data] {len(uni)} stocks, {len(ds):,} stock-days, {ds.shape[1]} columns, last day {ds['date'].max().date()}")
    return ds, uni, bench


def main(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .model import gpu_available, threads
    log(f"[env] {threads()} CPU threads, GPU {'yes' if gpu_available() else 'no'}; output folder {out.resolve()}")
    ds, uni, bench = load_all(a.tickers, a.refresh)
    recipes = default_recipes()
    if a.recipes:
        want = [x.strip() for x in a.recipes.split(",") if x.strip()]
        unknown = [w for w in want if w not in recipes]
        if unknown:
            raise SystemExit(f"unknown or unavailable recipes {unknown}; available: {sorted(recipes)}")
        recipes = {k: recipes[k] for k in dict.fromkeys(["base"] + want)}
    tests = TESTS if not getattr(a, "test_end", None) else {**TESTS, "Past year": (TESTS["Past year"][0], a.test_end)}
    val = VAL
    if getattr(a, "val", None):
        v = [x.strip() for x in a.val.split(",")]
        if len(v) != 2:
            raise SystemExit("--val must be 'start,end', e.g. 2023-01-03,2024-12-31")
        val = (v[0], v[1])
    if a.stage in ("select", "all"):
        run_select(ds, uni, out, recipes, val)
    if a.stage in ("final", "all"):
        sel = out / "select.json"
        if not sel.exists():
            raise SystemExit(f"{sel} not found: run the 'select' stage first")
        allu = None
        if getattr(a, "pit_check", False):
            from .data import get_universe, load_tickers
            allu = get_universe(load_tickers(a.all_tickers), refresh=a.refresh)
        chosen = json.loads(sel.read_text())
        if chosen["recipe"] not in recipes:
            raise SystemExit(f"the selected recipe '{chosen['recipe']}' is not available here (missing library?); available: {sorted(recipes)}")
        run_final(ds, uni, bench, out, chosen, recipes, tests, allu)
        log(f"[done] report: {out / 'report.md'}   chart: {out / 'falcon_v1_v2_v3.png'}")
