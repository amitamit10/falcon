import numpy as np
import pandas as pd

from test_pipeline import synth
from tradingbot import sectorflow as F


def fixtures():
    u = {f"T{i}.TA": synth(n=300, seed=i) for i in range(3)}
    idx = pd.bdate_range("2018-06-01", periods=500); rng = np.random.default_rng(0)
    ser = {t: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .01, 500))), index=idx) for t in set(F.SECTOR_ETF.values()) | set(F.MARKET_ETF.values())}
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, .008, 500))), index=idx)
    sec = pd.DataFrame({"ticker": list(u), "sector": ["Financial Services", "Technology", "Mystery"]})
    return u, ser, spx, sec


def test_each_stock_gets_its_own_sector_etf_and_unknown_sectors_stay_empty():
    u, ser, spx, sec = fixtures(); r = F.sector_flow(u, sec, ser, spx).set_index(["ticker", "date"])
    d = u["T0.TA"].index[100]
    assert r.loc[("T0.TA", d), "sector_us_ret_1d"] == ser["XLF"].pct_change().loc[d] and r.loc[("T1.TA", d), "sector_us_ret_1d"] == ser["XLK"].pct_change().loc[d]
    assert np.isclose(r.loc[("T0.TA", d), "sector_us_rel_1d"], ser["XLF"].pct_change().loc[d] - spx.pct_change().loc[d])
    assert r.loc["T2.TA", "sector_us_ret_1d"].isna().all() and r.loc["T2.TA", "gold_ret_1d"].notna().mean() > 0.9       # market factors still apply to every stock
    assert set(F.SECTOR_COLUMNS) <= set(r.columns)


def test_values_at_a_date_use_only_that_date_and_earlier():
    u, ser, spx, sec = fixtures(); d = u["T0.TA"].index[150]
    a = F.sector_flow(u, sec, ser, spx).set_index(["ticker", "date"]).loc[("T0.TA", d)]
    cut = {t: s.where(s.index <= d, s * 3) for t, s in ser.items()}; cs = spx.where(spx.index <= d, spx * 3)           # wreck everything after d
    b = F.sector_flow(u, sec, cut, cs).set_index(["ticker", "date"]).loc[("T0.TA", d)]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_offline_with_no_data_gives_empty_columns_and_feature_sets_exist():
    u, _, _, sec = fixtures(); empty = {t: pd.Series(dtype=float, index=pd.DatetimeIndex([])) for t in set(F.SECTOR_ETF.values()) | set(F.MARKET_ETF.values())}
    r = F.sector_flow(u, sec, empty, pd.Series(dtype=float, index=pd.DatetimeIndex([])))
    assert r[F.SECTOR_COLUMNS].isna().all().all() and len(r) == sum(len(v) for v in u.values())
    from tradingbot.model import feature_columns
    assert set(F.SECTOR_COLUMNS) <= set(feature_columns("xss")) and len(feature_columns("xssa")) > len(feature_columns("xsda"))
