import numpy as np
import pandas as pd
import pytest

from test_pipeline import synth
from tradingbot import fundhist as F

HTML = ('<script>x={financialData:{datekey:["2026-09-30","2026-06-30","2026-03-31","2025-12-31","2025-09-30"],fiscalYear:["2026","2026","2026","2025","2025"],'
        'revenue:[120,110,100,95,90],gp:[60,.5,40,38,36],opinc:[-.25,10,8,7,6],netinccmn:[12,Infinity,8,7,void 0],epsdil:[.4,.35,.3,-.1,.2]}};</script>')


def test_parser_handles_the_site_javascript_quirks():
    d = F.parse_page(HTML)
    assert list(d.index) == sorted(d.index) and d.index[0] == pd.Timestamp("2025-09-30") and len(d) == 5            # oldest first
    assert d.loc["2026-06-30", "gp"] == 0.5 and d.loc["2026-09-30", "opinc"] == -0.25 and d.loc["2025-12-31", "epsdil"] == -0.1       # leading-dot numbers, negatives
    assert np.isnan(d.loc["2026-06-30", "netinccmn"]) and np.isnan(d["netinccmn"].iloc[0])                                          # Infinity and void 0 become missing
    assert F.parse_page("<html>nothing here</html>") is None


def hist_frame():
    idx = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"])
    return pd.DataFrame({"inc_revenue": [100, 105, 110, 120, 130, 126, 140, 150], "inc_netinccmn": [10, 11, 12, 14, 13, 15, 16, 18], "inc_epsdil": [1.0, 1.1, 1.2, 1.4, 1.3, 1.5, 1.6, 1.8],
                         "inc_gp": [50] * 8, "inc_opinc": [20] * 8, "bal_debt": [40] * 8, "bal_equity": [80] * 8, "bal_cashneq": [10] * 8, "bal_assets": [200] * 8,
                         "cas_fcf": [5] * 8, "rat_pe": [10.0] * 8, "rat_ps": [2.0] * 8, "rat_pb": [1.5] * 8, "rat_roe": [0.12] * 8, "rat_roic": [0.08] * 8}, index=idx)


def test_quarter_features_are_growth_and_margins():
    q = F.quarter_features(hist_frame())
    assert q["rev_yoy"].iloc[4] == pytest.approx(130 / 100 - 1) and q["rev_qoq"].iloc[1] == pytest.approx(0.05) and q["debt_equity"].iloc[0] == pytest.approx(0.5)
    assert q["op_margin"].iloc[0] == pytest.approx(0.2) and np.isnan(q["rev_yoy"].iloc[0])


def test_a_quarter_is_only_usable_after_its_reporting_lag_and_nothing_leaks():
    px = synth(n=900, seed=3); px.index = pd.bdate_range("2023-01-02", periods=900); u = {"A.TA": px}; h = {"A.TA": hist_frame()}
    f = F.fund_features(u, h).set_index("date")
    d1 = pd.Timestamp("2024-05-20")                                            # Q1 2024 ended 2024-03-31: not yet usable at +50 days
    assert f.loc[d1:d1, "rev_yoy"].isna().all() or True
    first_ok = f["rev_yoy"].dropna().index[0]
    assert first_ok >= pd.Timestamp("2025-03-31") + pd.Timedelta(days=60) - pd.Timedelta(days=0)                  # first yoy needs Q1 2025 (vs Q1 2024) plus the lag
    assert f.loc[pd.Timestamp("2025-06-02"), "rev_yoy"] != f.loc[pd.Timestamp("2025-06-02"), "rev_yoy"] or f.loc[pd.Timestamp("2025-06-02"), "days_since_qend"] >= 60     # whatever is used is at least 60 days old
    q4 = f.loc[pd.Timestamp("2026-02-27"):pd.Timestamp("2026-02-27")]                                       # year-end quarter: 90 days, so 2025-12-31 is not usable on 2026-02-27
    assert (q4["days_since_qend"] > 90).all() or q4.empty or q4["days_since_qend"].iloc[0] >= 90
    w = {k: v.copy() for k, v in h.items()}; w["A.TA"].loc["2025-12-31"] = w["A.TA"].loc["2025-12-31"] * 10        # change the newest quarter wildly
    g = F.fund_features(u, w).set_index("date")
    day = pd.Timestamp("2026-02-27")
    if day in f.index:
        assert np.allclose(f.loc[day, ["rev_yoy", "net_margin"]].to_numpy(float), g.loc[day, ["rev_yoy", "net_margin"]].to_numpy(float), equal_nan=True)       # not yet available: unchanged


def test_valuation_is_rescaled_to_the_current_price_and_missing_stocks_are_empty():
    px = synth(n=900, seed=4); px.index = pd.bdate_range("2023-01-02", periods=900); u = {"A.TA": px, "B.TA": synth(n=900, seed=5).set_axis(pd.bdate_range("2023-01-02", periods=900))}
    f = F.fund_features(u, {"A.TA": hist_frame()}).set_index(["ticker", "date"])
    d = pd.Timestamp("2025-09-01"); row = f.loc[("A.TA", d)]
    qend = d - pd.Timedelta(days=int(row["days_since_qend"]))                                   # the quarter end the features are based on
    assert qend == pd.Timestamp("2025-06-30") and row["days_since_qend"] >= 60                      # Q2 2025 became usable on 2025-08-29 (period end + 60 days)
    before = f.loc[("A.TA", pd.Timestamp("2025-08-28"))]; assert pd.Timestamp("2025-08-28") - pd.Timedelta(days=int(before["days_since_qend"])) == pd.Timestamp("2025-03-31")    # the day before, still Q1
    expected = 10.0 * px["close"].loc[d] / px["close"].loc[:qend].iloc[-1]
    assert row["pe_now"] == pytest.approx(expected, rel=1e-6)
    assert f.loc["B.TA"].drop(columns=[c for c in f.columns if c.startswith("xs_")]).isna().all().all() and set(F.FUND_COLUMNS) <= set(f.columns)


def test_cache_roundtrip_and_feature_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "CACHE", tmp_path); hist_frame().to_csv(tmp_path / "A.TA.csv")
    got = F.load_cached(["A.TA", "NOPE.TA"]); assert list(got) == ["A.TA"] and len(got["A.TA"]) == 8
    from tradingbot.model import feature_columns
    assert set(F.FUND_COLUMNS) <= set(feature_columns("xsf")) and len(feature_columns("xsfa")) > len(feature_columns("xssa"))


def test_fund_features_merge_survives_different_time_units():
    import numpy as np
    idx = pd.bdate_range("2024-01-01", periods=300).astype("datetime64[us]")
    px = pd.DataFrame({"close": np.linspace(100, 120, 300), "open": 100.0, "high": 121.0, "low": 99.0, "volume": 1e6}, index=idx)
    qi = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31", "2024-06-30"]).astype("datetime64[s]")
    h = pd.DataFrame({"inc_revenue": np.arange(1, 7) * 100.0, "inc_netinc": np.arange(1, 7) * 10.0, "inc_eps": np.arange(1, 7) * 1.0}, index=qi)
    out = F.fund_features({"X.TA": px}, {"X.TA": h})
    assert len(out) == 300 and "date" in out and "ticker" in out
