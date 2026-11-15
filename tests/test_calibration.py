import numpy as np
import pandas as pd

from tradingbot.calibration import Calibration, fit_calibration, reliability_table, wilson
from tradingbot.forward import brier, ece, evaluate, oos_predictions


def synthetic_oos(n_days=300, n=60, skill=0.25, seed=0, overconfident=True):
    """Scores that carry a little real signal, but are over-confident (spread too wide) - like an uncalibrated model."""
    rng = np.random.default_rng(seed); rows = []
    for d in pd.bdate_range("2023-01-02", periods=n_days):
        true_p = np.clip(0.5 + skill * rng.normal(0, 0.25, n), 0.1, 0.9)
        y = (rng.random(n) < true_p).astype(float)
        raw = 0.5 + (true_p - 0.5) * (2.5 if overconfident else 1.0)
        rows.append(pd.DataFrame({"date": d, "ticker": range(n), "p": np.clip(raw, 0.01, 0.99), "y": y, "ret": y - 0.5}))
    return pd.concat(rows, ignore_index=True)


def test_wilson_and_metrics():
    lo, hi = wilson(60, 100)
    assert 0.5 < lo < 0.6 < hi < 0.7
    assert brier([1, 0], [1, 0]) == 0 and abs(brier([0.5, 0.5], [1, 0]) - 0.25) < 1e-12
    p = np.linspace(0.05, 0.95, 5000); y = (np.random.default_rng(0).random(5000) < p).astype(float)
    assert ece(p, y) < 0.03                                    # honest probabilities have tiny calibration error


def test_calibration_fixes_overconfidence_on_holdout():
    oos = synthetic_oos()
    cal = fit_calibration(oos, min_hit=0.58, min_n=200)
    assert cal.stats["ece_calibrated"] < cal.stats["ece_raw"]                  # calibrated is closer to reality than raw
    assert cal.stats["brier_calibrated"] <= cal.stats["brier_raw"] + 1e-9
    rt = pd.DataFrame(cal.table)
    assert (rt["n"] > 0).all() and ((rt["actual_hit_rate"] - rt["mean_predicted"]).abs() < 0.08).all()   # 70% means ~70%


def test_no_signal_means_no_confident_predictions():
    oos = synthetic_oos(skill=0.0, overconfident=True)                         # pure noise
    cal = fit_calibration(oos, min_hit=0.55, min_n=300)
    assert cal.t_up is None and cal.t_down is None                             # nothing is flagged: never invent confidence
    assert (cal.flag(cal.apply(oos["p"])) == "").all() and cal.stats["max_calibrated_p"] < 0.56


def test_real_signal_gets_a_validated_threshold_and_roundtrip():
    cal = fit_calibration(synthetic_oos(skill=0.9), min_hit=0.60, min_n=200)
    assert cal.t_up is not None
    flags = cal.flag(cal.apply([0.5, 0.99]))
    assert flags[0] == "" and flags[1] == "up"
    again = Calibration.from_json(cal.to_json())
    assert again.t_up == cal.t_up and np.allclose(again.apply([0.3, 0.7]), cal.apply([0.3, 0.7]))


def test_oos_predictions_and_evaluate_on_pipeline_data():
    import sys; sys.path.insert(0, "tests")
    from test_pipeline import synth
    from tradingbot.model import build_dataset
    ds = build_dataset({f"T{i}.TA": synth(n=800, seed=i) for i in range(8)}, horizon=10)
    oos = oos_predictions(ds, "2022-01-01", "2022-12-31", retrain_every=63, params=dict(max_iter=20))
    assert set(oos.columns) == {"date", "ticker", "p", "y", "ret"} and oos["p"].between(0, 1).all()
    ev = evaluate(oos, horizon=1, min_stocks=5)
    assert 0.4 < ev["auc"] < 0.6 and ev["n"] == len(oos)                      # random walk: no skill, no leak


def test_calibration_cannot_claim_more_than_a_whole_bucket_achieved():
    """A handful of extreme, lucky scores must not turn into a 98% claim (the isotonic tail-overfit we hit on real data)."""
    rng = np.random.default_rng(1)
    n = 40000
    p = rng.normal(0.5, 0.03, n)
    y = (rng.random(n) < 0.5).astype(float)                    # pure noise
    p[:30] = 0.99; y[:30] = 1.0                                # 30 lucky extreme points
    oos = pd.DataFrame({"date": np.repeat(pd.bdate_range("2023-01-02", periods=400), n // 400), "ticker": 0, "p": p, "y": y, "ret": 0.0})
    cal = fit_calibration(oos, min_hit=0.55, min_n=100)
    assert cal.stats["max_calibrated_p"] < 0.60 and cal.stats["min_calibrated_p"] > 0.40
    assert float(cal.apply([0.99])[0]) < 0.60                    # even a 0.99 raw score cannot be mapped to a huge probability
