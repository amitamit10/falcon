import numpy as np
import pandas as pd
import pytest

from tradingbot import indicators as ind
from tradingbot.features import FEATURE_COLUMNS, build_features
from tradingbot.model import build_dataset


def synth(n=1200, seed=0, drift=0.0004):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.015, n)
    c = 100 * np.exp(np.cumsum(r))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(c, o) * (1 + abs(rng.normal(0, 0.005, n)))
    l = np.minimum(c, o) * (1 - abs(rng.normal(0, 0.005, n)))
    v = rng.integers(1_000, 5_000, n).astype(float)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v},
                        index=pd.bdate_range("2019-01-01", periods=n))


def test_rsi_bounds_and_trend():
    s = pd.Series(np.arange(1, 100, dtype=float))
    assert ind.rsi(s).dropna().iloc[-1] == 100
    assert ind.rsi(synth()["close"]).dropna().between(0, 100).all()


def test_no_lookahead_in_features():
    df = synth()
    full = build_features(df)
    cut = build_features(df.iloc[:800])
    pd.testing.assert_frame_equal(full.iloc[:800], cut, check_exact=False, rtol=1e-9)


def test_breakout_flag():
    df = synth(300)
    df.iloc[-1, df.columns.get_loc("close")] = df["high"].max() * 1.1
    assert build_features(df)["breakout_20"].iloc[-1] == 1.0


@pytest.fixture(scope="module")
def ds():
    return build_dataset({f"T{i}.TA": synth(seed=i) for i in range(5)})


def test_dataset_rows_only_use_data_known_at_the_close(ds):
    """Features at date t must not change when later bars change (the dataset feeds the scan, the technical read-out and the rules)."""
    base = {f"T{i}.TA": synth(seed=i) for i in range(3)}
    cut = base["T0.TA"].index[700]
    wrecked = {k: v.copy() for k, v in base.items()}
    for v in wrecked.values():
        v.loc[v.index > cut, ["open", "high", "low", "close"]] *= 3
    a = build_dataset(base); b = build_dataset(wrecked)
    cols = ["rsi14", "dist_ma50", "dist_ma150", "rule_score", "atr_pct"]
    x = a[a["date"] == cut].set_index("ticker")[cols]; y = b[b["date"] == cut].set_index("ticker")[cols]
    assert np.allclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True)


def test_feature_columns_present(ds):
    assert set(FEATURE_COLUMNS) <= set(ds.columns)


def test_bad_bar_is_not_a_label_and_rule_score_has_no_nan():
    df = synth(400)
    df.iloc[300:, df.columns.get_loc("close")] *= 500  # glitch: 500x jump
    d = build_dataset({"X.TA": df})
    assert d["next_ret"].abs().max() <= 1.0
    assert d["rule_score"].notna().all()


def test_pivot_resistance_breakout_flag():
    n = 120
    c = np.concatenate([np.linspace(100, 110, 40), np.linspace(110, 100, 40), np.linspace(100, 125, 40)])
    df = pd.DataFrame({"open": c, "high": c + 0.2, "low": c - 0.2, "close": c, "volume": 1000.0},
                      index=pd.bdate_range("2023-01-01", periods=n))
    f = build_features(df)
    assert f["sr_breakout"].sum() >= 1
    assert (f["pivot_res"].dropna() >= 109).any()
