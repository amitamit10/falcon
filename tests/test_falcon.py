import json

import numpy as np
import pandas as pd
import pytest

from test_pipeline import synth
from tradingbot import falcon as F
from tradingbot.model import build_dataset

VAL = ("2021-06-01", "2021-09-30")
TESTS = {"Early": ("2021-10-01", "2021-12-31"), "Late": ("2022-01-03", "2022-05-31")}


@pytest.fixture(autouse=True)
def few_stocks(monkeypatch):
    monkeypatch.setattr(F, "MIN_STOCKS", 4)                              # the synthetic world has 6 stocks, not 117


@pytest.fixture(scope="module")
def world():
    uni = {f"T{i}.TA": synth(n=900, seed=i, drift=0.0006) for i in range(6)}
    ds = build_dataset(uni, horizon=10)
    return ds, uni, uni["T0.TA"]["close"]


def tiny_recipes():
    M = F.Member
    base = F.Recipe("base", "tiny base", (M("1d", "hgb", "xs", window=3), M("10d", "hgb", "xs", window=3)))
    alt = F.Recipe("alt", "base plus a rank regression", base.members + (M("10d", "lgbm_reg", "xs", target="rank_target", window=3),))
    return {"base": base, "alt": alt}


def test_member_keys_and_targets():
    m, r = F.Member("10d", "lgbm_rank", "xsx", target="rank_q"), F.Member("1d", "lgbm", "xs")
    assert m.date_col == "label_date" and not m.is_classifier and r.date_col == "label_date1" and r.is_classifier
    assert r.key != F.Member("1d", "lgbm", "xs", every=10).key != F.Member("1d", "lgbm", "xs", window=None).key       # every setting changes the cache key
    assert F.Member("10d", "hgb", "xsx").tgt == "label_xs"


def test_combine_averages_probabilities_or_ranks():
    d = pd.Timestamp("2026-01-05"); t = ["A", "B", "C"]
    a = pd.DataFrame({"date": d, "ticker": t, "p": [.6, .5, .4]}); b = pd.DataFrame({"date": d, "ticker": t, "p": [.4, .5, .6]})
    c = F.combine([a, b], [F.Member("1d", "lgbm", "xs")] * 2).set_index("ticker")["p"]
    assert c.to_dict() == pytest.approx({"A": .5, "B": .5, "C": .5})                                           # probabilities averaged
    reg = pd.DataFrame({"date": d, "ticker": t, "p": [10.0, 20.0, 30.0]})                                       # a different scale: only ranks may be mixed
    r = F.combine([a, reg], [F.Member("10d", "hgb", "xs"), F.Member("10d", "lgbm_reg", "xs", target="rank_target")]).set_index("ticker")["p"]
    assert r.to_dict() == pytest.approx({"A": (1.0 + 1 / 3) / 2, "B": (2 / 3 + 2 / 3) / 2, "C": (1 / 3 + 1.0) / 2})
    short = pd.DataFrame({"date": d, "ticker": ["A", "B"], "p": [.1, .2]})
    assert set(F.combine([a, short], [F.Member("1d", "lgbm", "xs")] * 2)["ticker"]) == {"A", "B"}                 # only stock-days every member predicted


def test_checkpointing_resumes_and_never_repeats_work(world, tmp_path, monkeypatch):
    ds, _, _ = world
    import tradingbot.model as model
    calls = []
    real = model.train
    monkeypatch.setattr(model, "train", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    m = F.Member("1d", "hgb", "xs", window=3)
    first = F.member_predictions(ds, m, *VAL, tmp_path)
    n1 = len(calls); assert n1 >= 4 and first["p"].between(0, 1).all() and set(first.columns) == {"date", "ticker", "p"}
    again = F.member_predictions(ds, m, *VAL, tmp_path)
    assert len(calls) == n1 and len(again) == len(first)                                                         # second run: everything came from disk
    files = sorted((tmp_path / "members" / m.key / f"{VAL[0]}_{VAL[1]}").glob("*.csv.gz"))
    files[-1].unlink(); F.member_predictions(ds, m, *VAL, tmp_path); assert len(calls) == n1 + 1                # an interrupted run redoes only the missing block
    assert not list((tmp_path / "members").rglob("*.tmp"))                                                       # no half-written files left
    F.member_predictions(ds, F.Member("1d", "hgb", "xs", every=10, window=3), *VAL, tmp_path); assert len(calls) > n1 + 1      # a changed setting is a new cache entry


def test_predictions_use_only_labels_known_at_the_block_start(world, tmp_path):
    ds, _, _ = world
    m = F.Member("10d", "hgb", "xs", window=3); p = F.member_predictions(ds, m, *VAL, tmp_path).set_index(["date", "ticker"])["p"]
    cut = pd.Timestamp("2021-08-02")
    poisoned = ds.copy(); poisoned.loc[poisoned["date"] > cut, ["label_xs", "excess_ret", "rank_target"]] = 1.0           # future outcomes changed
    q = F.member_predictions(poisoned, m, *VAL, tmp_path / "poison").set_index(["date", "ticker"])["p"]
    early = [i for i in p.index if i[0] < cut - pd.Timedelta(days=30)]
    assert np.allclose(p.loc[early], q.loc[early])                                                               # predictions before the poison cannot depend on it


def test_signal_quality_and_portfolio_search(world, tmp_path, monkeypatch):
    ds, uni, _ = world
    monkeypatch.setattr(F, "GRID", dict(n=(2, 3), dm=(2,), stop=(2.0, 3.0)))
    r = tiny_recipes()["base"]; p1, p10 = F.recipe_signals(ds, r, *VAL, tmp_path)
    q = F.signal_quality(ds, p1, p10, *VAL)
    assert 0.3 < q["1d"]["auc"] < 0.7 and q["score"] == pytest.approx(F.W10 * q["10d"]["ic"] + (1 - F.W10) * q["1d"]["ic"])
    st, grid = F.search_portfolio(p1, p10, uni, *VAL)
    assert st.name == "Falcon v3" and st.top_n in (2, 3) and len(grid) == 4 and grid["sharpe"].is_monotonic_decreasing


def test_scenarios_have_the_promised_frictions():
    from tradingbot.strategies import FALCON_V2
    assert F.SCENARIOS == ("Ideal", "Real-world", "Real-world, order during the day", "Real-world, one day late (stress test)")
    ideal, real, during, slow = (F.make_params(FALCON_V2, s) for s in F.SCENARIOS)
    assert ideal.spread == 0 and ideal.unit == 1.0 and ideal.delay_days == 0
    assert real.spread > 0 and real.max_adv_frac > 0 and real.whole_shares and real.tax_rate == .25 and real.unit == .01 and real.capital == 1_000_000 and real.delay_days == 0
    assert during.delay_days == 0 and during.fill == "ohlc4" and during.spread == real.spread and real.fill == "open"          # same morning, but filled at the day's average price
    assert slow.delay_days == 1 and slow.fill == "ohlc4" and slow.spread == real.spread
    assert real.max_pos == FALCON_V2.top_n and real.stop_atr == FALCON_V2.stop_atr
    with pytest.raises(ValueError):
        F.make_params(FALCON_V2, "nope")


def test_point_in_time_membership_follows_liquidity():
    idx = pd.bdate_range("2021-01-04", periods=200)
    def frame(vol): return pd.DataFrame({"close": 100.0, "volume": vol}, index=idx)
    allu = {f"S{i}.TA": frame(1000.0 * (i + 1)) for i in range(6)}
    allu["S0.TA"] = frame(np.r_[np.full(100, 9e6), np.full(100, 1.0)])                       # S0 was the most liquid, then dried up
    m = F.pit_members(allu, top=2)
    early, late = idx[90], idx[-1]
    assert (early, "S0.TA") in m and (late, "S0.TA") not in m and (late, "S5.TA") in m and (late, "S4.TA") in m
    p = pd.DataFrame({"date": [early, late, late], "ticker": ["S0.TA", "S0.TA", "S5.TA"], "p": [.5, .5, .5]})
    assert len(F._filter(p, m)) == 2


def test_select_then_final_end_to_end(world, tmp_path, monkeypatch):
    ds, uni, bench = world
    monkeypatch.setattr(F, "GRID", dict(n=(2, 3), dm=(2,), stop=(2.0,)))
    recipes = tiny_recipes()
    ch = F.run_select(ds, uni, tmp_path, recipes, VAL)
    assert ch["recipe"] in recipes and {r["recipe"] for r in ch["table"]} == set(recipes) and ch["base_reference"]["strategy"]
    assert ch["table"][0]["score"] >= ch["table"][1]["score"] and (tmp_path / "select.json").exists() and (tmp_path / "select_recipes.csv").exists()
    out = F.run_final(ds, uni, bench, tmp_path, json.loads((tmp_path / "select.json").read_text()), recipes, TESTS, allu=uni)
    assert set(out["results"]) == set(TESTS)
    for period in TESTS:
        d = out["results"][period]
        assert {"benchmarks", *F.SCENARIOS, "Real-world, point-in-time universe"} <= set(d) and set(d["Ideal"]) == {"Falcon v1", "Falcon v2", "Falcon v3"}
        assert np.isfinite(d["Ideal"]["Falcon v2"]["return"]) and d["Real-world"]["Falcon v1"]["start_equity"] == 1_000_000
        assert np.isfinite(d["benchmarks"]["buy_and_hold"]) and np.isfinite(d["benchmarks"]["index"])
    assert (tmp_path / "final.json").exists() and (tmp_path / "falcon_v1_v2_v3.png").read_bytes()[:4] == b"\x89PNG" and list(tmp_path.glob("trades_*.csv"))
    rep = (tmp_path / "report.md").read_text()
    assert "## Early" in rep and "### Real-world, one day late" in rep and "| Falcon v3 |" in rep and "Rank IC" in rep and "paper money" in rep.lower()
    again = F.run_select(ds, uni, tmp_path, recipes, VAL)
    assert again["recipe"] == ch["recipe"]                                                                      # a rerun resumes from the cache with the same choice


def test_cli_stage_checks_and_unknown_recipe(world, tmp_path, monkeypatch):
    import argparse
    ds, uni, bench = world
    monkeypatch.setattr(F, "load_all", lambda *a, **k: (ds, uni, bench))
    a = argparse.Namespace(out=str(tmp_path), tickers="x", refresh=False, recipes="nope", stage="select", pit_check=False, all_tickers="x", test_end=None)
    with pytest.raises(SystemExit, match="unknown or unavailable"):
        F.main(a)
    a.recipes = ""; a.stage = "final"
    with pytest.raises(SystemExit, match="select"):
        F.main(a)
    names = F.default_recipes()
    assert {"base", "fast10", "ens5", "cross", "longwin"} <= set(names) and all(m.every in (10, 21) for r in names.values() for m in r.members)
