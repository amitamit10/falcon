import numpy as np
import pandas as pd
import pytest

from test_pipeline import synth
from tradingbot import dual, macro
from tradingbot.alpha158 import ALPHA_COLUMNS, alpha_features
from tradingbot.extra import EXTRA_COLUMNS, add_extra, stock_extra
from tradingbot.features_more import augment
from tradingbot.model import build_dataset, feature_columns, predict, train


@pytest.fixture(scope="module")
def uni():
    return {f"T{i}.TA": synth(n=900, seed=i) for i in range(6)}


@pytest.fixture(scope="module")
def ds(uni):
    return build_dataset(uni, horizon=10)


def wrecked(uni, cut):
    out = {}
    for k, v in uni.items():
        w = v.copy(); w.loc[w.index > cut, ["open", "high", "low", "close", "volume"]] *= 3; out[k] = w
    return out


def test_alpha158_and_extra_features_never_look_ahead(uni):
    a = uni["T0.TA"]; cut = a.index[600]; w = wrecked(uni, cut)["T0.TA"]
    for fn, cols in ((alpha_features, ALPHA_COLUMNS), (stock_extra, None)):
        x, y = fn(a), fn(w)
        cc = cols or list(x.columns)
        assert np.allclose(x.loc[cut, cc].to_numpy(float), y.loc[cut, cc].to_numpy(float), equal_nan=True), fn.__name__
    assert alpha_features(a).shape[1] == len(ALPHA_COLUMNS) == 157


def test_extra_features_values_and_sector_context(uni, ds):
    d = uni["T0.TA"]; e = stock_extra(d)
    assert np.isclose(e["gap_open"].iloc[10], d["open"].iloc[10] / d["close"].iloc[9] - 1) and e["close_in_range"].dropna().between(0, 1).all()
    st = e["streak"].iloc[1:]; assert (st != 0).all() and st.abs().max() >= 2          # the first bar has no previous close
    sectors = pd.DataFrame({"ticker": list(uni), "sector": ["A", "A", "A", "A", "B", "B"]})
    out = add_extra(ds, uni, sectors)
    day = out[out["date"] == out["date"].max()].set_index("ticker")
    assert day.loc["T0.TA", "sector_ret_1d"] == pytest.approx(day.loc[["T0.TA", "T1.TA", "T2.TA", "T3.TA"], "ret_1d"].mean())
    assert day.loc["T4.TA", "sector_ret_1d"] != day.loc["T4.TA", "sector_ret_1d"]            # a sector of 2 stocks (< 4) is left empty
    assert set(EXTRA_COLUMNS) <= set(out.columns) and out["dow"].between(0, 4).all()


def test_macro_features_use_only_their_own_day_and_survive_being_offline(monkeypatch, tmp_path, uni, ds):
    idx = pd.bdate_range("2018-06-01", periods=1300)
    series = {t: pd.Series(100 + np.arange(1300, dtype=float) + k, index=idx) for k, t in enumerate(macro.SERIES.values())}
    monkeypatch.setattr(macro, "_close", lambda t, start, refresh: series[t])
    m = macro.macro_frame()
    gspc = series["^GSPC"]
    assert m.loc[idx[10], "us_ret_1d"] == pytest.approx(gspc.iloc[10] / gspc.iloc[9] - 1) and set(macro.MACRO_COLUMNS) <= set(m.columns)
    both = macro.add_macro(ds, m)
    assert both.shape[0] == ds.shape[0] and both["us_ret_1d"].notna().mean() > 0.9
    monkeypatch.setattr(macro, "_close", lambda t, start, refresh: pd.Series(dtype=float, index=pd.DatetimeIndex([])))      # offline, no cache
    e = macro.add_macro(ds, macro.macro_frame())
    assert e["us_ret_1d"].isna().all() and e.shape[0] == ds.shape[0]
    monkeypatch.setattr(macro, "CACHE", tmp_path / "mc")


def test_augment_adds_every_family_and_models_still_train(uni, ds, monkeypatch):
    monkeypatch.setattr(macro, "_close", lambda t, start, refresh: pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    big = augment(ds, uni, sectors_path="no/such/file.csv")
    assert set(ALPHA_COLUMNS) <= set(big.columns) and set(EXTRA_COLUMNS) <= set(big.columns) and set(macro.MACRO_COLUMNS) <= set(big.columns)
    asof = pd.Timestamp(big["date"].max()) - pd.Timedelta(days=30)
    for fs in ("xs", "xsm", "xse", "xsx", "xsa", "xsxa"):
        m = train(big, asof, 3, target="label_up1", date_col="label_date1", algo="lgbm", feature_set=fs)
        p = predict(m, big[big["date"] == big["date"].max()].dropna(subset=["rsi14"]))
        assert p.between(0, 1).all(), fs
    assert len(feature_columns("xsxa")) > len(feature_columns("xsx")) > len(feature_columns("xs"))


def test_other_training_objectives_run(ds):
    asof = pd.Timestamp(ds["date"].max()) - pd.Timedelta(days=30); rows = ds[ds["date"] == ds["date"].max()].dropna(subset=["rsi14"])
    for target, algo in (("ret_fwd1", "lgbm_reg"), ("rank_target", "lgbm_reg"), ("rank_q", "lgbm_rank")):
        date_col = "label_date1" if target == "ret_fwd1" else "label_date"
        m = train(ds, asof, 3, target=target, date_col=date_col, algo=algo)
        assert np.isfinite(predict(m, rows)).all(), (target, algo)


def test_ensemble_averages_members_and_signature_tracks_the_recipe(ds, monkeypatch):
    sig = dual.signature("1d")
    assert sig == dual.signature("1d") and sig != dual.signature("10d")
    old = dual.SPECS["1d"]["members"]
    monkeypatch.setitem(dual.SPECS["1d"], "members", old[:2])
    assert dual.signature("1d") != sig                                               # a changed recipe never reuses an old calibration
    calls = []
    def fake(ds_, start, end, target, date_col, ret_col, retrain_every=63, window_years=6, **kw):
        calls.append(kw["feature_set"]); i = len(calls)
        base = ds_[ds_["date"].between(start, end)][["date", "ticker"]].copy(); base["p"] = 0.4 + 0.1 * i; base["y"] = 1.0; base["ret"] = 0.0
        return base
    monkeypatch.setattr(dual, "oos_predictions", fake)
    o = dual.ensemble_oos(ds, "2023-06-01", "2023-08-01", "1d")
    assert calls == ["xs", "xsx"] and np.allclose(o["p"], 0.55)                       # mean of 0.5 and 0.6


def test_pro_setup_features_are_point_in_time_and_sensible():
    from test_pipeline import synth
    from tradingbot import pro
    uni = {f"T{i}.TA": synth(n=400, seed=i) for i in range(4)}
    closes = pd.concat({tk: d["close"] for tk, d in uni.items()}, axis=1)
    mkt = (1 + closes.pct_change().mean(axis=1).fillna(0)).cumprod()
    f = pro.pro_features(uni["T0.TA"], mkt)
    assert set(pro.PRO_COLUMNS) <= set(f.columns) and f["trend_template"].dropna().between(0, 6).all() and f["brk_20d"].dropna().isin([0.0, 1.0]).all()
    cut = f.index[300]; poisoned = uni["T0.TA"].copy(); poisoned.loc[poisoned.index > cut, ["open", "high", "low", "close", "volume"]] *= 3
    g = pro.pro_features(poisoned, mkt)
    assert np.allclose(f.loc[:cut].fillna(-9).values, g.loc[:cut].fillna(-9).values)            # nothing after `cut` can change a value on or before it
    from tradingbot.model import feature_columns
    assert feature_columns("xsp")[-len(pro.PRO_COLUMNS):] == pro.PRO_COLUMNS and set(feature_columns("xss")) < set(feature_columns("xspa"))


def test_gauss_targets_are_standardised_and_sector_neutral():
    from tradingbot.features_more import add_gauss_targets
    d = pd.date_range("2024-01-01", periods=3)
    rows = [dict(date=x, ticker=f"S{i}", next_ret=(i + 1) * (1 if j % 2 else -1) * .01) for j, x in enumerate(d) for i in range(8)]
    ds = pd.DataFrame(rows); sec = pd.DataFrame({"ticker": [f"S{i}" for i in range(8)], "sector": ["A"] * 4 + ["B"] * 4})
    out = add_gauss_targets(ds, sec)
    g = out.groupby("date")["rk_gauss"]
    assert np.allclose(g.mean(), 0, atol=1e-9)
    assert out.groupby(["date", out["ticker"].map(dict(zip(sec["ticker"], sec["sector"])))])["sn_gauss"].mean().abs().max() < 0.6 and out[["rk_gauss", "sn_gauss"]].notna().all().all()
