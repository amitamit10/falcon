"""Classic swing-trading features (trend, breakout, pullback, volume). Every value at row t uses only data up to and including t.

Rules encoded (from the strategy notes):
- Trend filter: price above a rising/flat MA150; higher highs / higher lows.
- Distance from MA100/150/200 (momentum), pull-back to MA20/MA50 for entry timing.
- Golden/death cross (20/50, 50/200).
- Resistance breakout (close above prior N-day high) with volume confirmation.
- Consolidation with steady volume (tight range + volume not falling).
- RSI / CCI / MACD / Stochastic / Bollinger (squeeze, %B).
- Candles: hammer, shooting star, engulfing, doji.
- Relative strength vs benchmark (TA-125 / S&P) when supplied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind

FEATURE_COLUMNS: list[str] = []  # filled at bottom


def _slope(s: pd.Series, n: int = 10) -> pd.Series:
    return s / s.shift(n) - 1


def build_features(df: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """df: index=date, columns open/high/low/close/volume. Returns feature frame."""
    c, o, h, l, v = df["close"], df["open"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    ma20, ma50, ma100, ma150, ma200 = (ind.sma(c, n) for n in (20, 50, 100, 150, 200))
    for n, m in ((20, ma20), (50, ma50), (100, ma100), (150, ma150), (200, ma200)):
        f[f"dist_ma{n}"] = c / m - 1
    f["ma150_slope"] = _slope(ma150)
    f["ma200_slope"] = _slope(ma200)
    f["above_ma150"] = (c > ma150).astype(float).where(ma150.notna())
    f["trend_ok"] = ((c > ma150) & (f["ma150_slope"] >= -0.005)).astype(float).where(ma150.notna())
    f["cross_20_50"] = (ma20 > ma50).astype(float).where(ma50.notna())
    f["cross_50_200"] = (ma50 > ma200).astype(float).where(ma200.notna())
    f["golden_cross_recent"] = ((ma20 > ma50) & (ma20.shift(5) <= ma50.shift(5))).astype(float).where(ma50.notna())
    # pullback to MA20/50 inside an uptrend = optimal entry timing
    f["pullback_entry"] = (((c / ma20 - 1).abs() < 0.015) | ((c / ma50 - 1).abs() < 0.02)).astype(float) * f["trend_ok"]

    # higher highs / higher lows over two consecutive 20-day windows
    hh = h.rolling(20).max() > h.rolling(20).max().shift(20)
    hl = l.rolling(20).min() > l.rolling(20).min().shift(20)
    f["higher_highs"] = hh.astype(float).where(h.shift(40).notna())
    f["higher_lows"] = hl.astype(float).where(l.shift(40).notna())

    # resistance breakout + volume
    prior_high = h.rolling(20).max().shift(1)
    prior_low = l.rolling(20).min().shift(1)
    vol_ratio = v / v.rolling(20, min_periods=20).mean()
    f["vol_ratio"] = vol_ratio
    f["breakout_20"] = (c > prior_high).astype(float).where(prior_high.notna())
    f["breakdown_20"] = (c < prior_low).astype(float).where(prior_low.notna())
    f["breakout_vol"] = f["breakout_20"] * (vol_ratio > 1.3).astype(float)
    f["dist_20d_high"] = c / h.rolling(20).max() - 1
    f["dist_52w_high"] = c / h.rolling(252, min_periods=120).max() - 1

    # consolidation: tight range, volume not falling
    rng = (h.rolling(15).max() - l.rolling(15).min()) / c
    f["range_15d"] = rng
    f["consolidation_vol"] = ((rng < rng.rolling(120, min_periods=60).median()) &
                              (v.rolling(5).mean() >= v.rolling(15).mean())).astype(float)

    f["rsi14"] = ind.rsi(c)
    f["rsi_slope"] = f["rsi14"].diff(3)
    f["cci20"] = ind.cci(df)
    m, s, hist = ind.macd(c)
    f["macd_hist_pct"] = hist / c
    f["macd_above_signal"] = (m > s).astype(float).where(s.notna())
    f["macd_hist_change"] = hist.diff() / c
    pk, pd_ = ind.stochastic(df)
    f["stoch_k"], f["stoch_d"] = pk, pd_
    mid, up, lo = ind.bollinger(c)
    bw = (up - lo) / mid
    f["boll_pctb"] = (c - lo) / (up - lo).replace(0, np.nan)
    f["boll_width"] = bw
    f["boll_squeeze"] = (bw < bw.rolling(120, min_periods=60).quantile(0.2)).astype(float).where(bw.notna())
    f["atr_pct"] = ind.atr(df) / c
    f["vwma_dist"] = c / ind.vwma(c, v) - 1

    # slide 15 step 1: weekly view vs MA100 (weekly MA100 ~ 500 trading days); farther = more momentum
    f["dist_wma100"] = c / ind.sma(c, 500) - 1

    # slide 13: divergence (price new low/high over 20d but indicator does not confirm)
    def _div(series: pd.Series):
        p_ll = c.rolling(20).min() < c.rolling(20).min().shift(20)
        p_hh = c.rolling(20).max() > c.rolling(20).max().shift(20)
        i_hl = series.rolling(20).min() > series.rolling(20).min().shift(20)
        i_lh = series.rolling(20).max() < series.rolling(20).max().shift(20)
        ok = series.shift(40).notna()
        return (p_ll & i_hl).astype(float).where(ok), (p_hh & i_lh).astype(float).where(ok)
    f["rsi_bull_div"], f["rsi_bear_div"] = _div(f["rsi14"])
    f["macd_bull_div"], f["macd_bear_div"] = _div(hist)

    # slide 11: Stochastic RSI (stochastic formula applied to RSI)
    r = f["rsi14"]
    f["stoch_rsi"] = (r - r.rolling(14).min()) / (r.rolling(14).max() - r.rolling(14).min()).replace(0, np.nan)

    # slide 20: stop distance = 2 ATR below price (risk sizing feature)
    f["stop_dist_2atr"] = 2 * ind.atr(df) / c

    # support / resistance from CONFIRMED pivots (a pivot at t-k needs k bars to its right, so it is known at t)
    k = 5
    ph = h.where(h == h.rolling(2 * k + 1, center=True).max()).shift(k)
    pl = l.where(l == l.rolling(2 * k + 1, center=True).min()).shift(k)
    res = ph.ffill(limit=120); sup = pl.ffill(limit=120)
    f["pivot_res"], f["pivot_sup"] = res, sup
    f["dist_res"] = res / c - 1          # >0: resistance still above price
    f["dist_sup"] = c / sup - 1          # >0: support below price
    # resistance that price has just closed above for the first time = a breakout (becomes support)
    f["sr_breakout"] = ((c > res) & (c.shift(1) <= res)).astype(float).where(res.notna())

    from .patterns import all_patterns
    f = pd.concat([f, all_patterns(df).drop(columns=["trendline", "cup_target"])], axis=1)

    # candles
    body = (c - o).abs()
    rngc = (h - l).replace(0, np.nan)
    upper = h - np.maximum(c, o)
    lower = np.minimum(c, o) - l
    f["candle_body_pct"] = (c - o) / rngc
    f["hammer"] = ((lower > 2 * body) & (upper < body.clip(lower=1e-9)) & (f["dist_ma20"] < 0)).astype(float)
    f["shooting_star"] = ((upper > 2 * body) & (lower < body.clip(lower=1e-9)) & (f["dist_ma20"] > 0)).astype(float)
    f["doji"] = (body / rngc < 0.1).astype(float)
    po, pc = o.shift(1), c.shift(1)
    f["bull_engulf"] = ((pc < po) & (c > o) & (c >= po) & (o <= pc)).astype(float)
    f["bear_engulf"] = ((pc > po) & (c < o) & (c <= po) & (o >= pc)).astype(float)

    # returns / momentum
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}d"] = c.pct_change(n)
    f["gap"] = o / c.shift(1) - 1

    # --- extra momentum / risk features (stage 2) ---
    r1 = c.pct_change()
    f["ret_2d"] = c.pct_change(2)
    f["ret_60d"], f["ret_120d"] = c.pct_change(60), c.pct_change(120)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1          # classic 12-1 month momentum (skip the last month)
    f["vol_20d"], f["vol_60d"] = r1.rolling(20).std(), r1.rolling(60).std()
    f["vol_ratio_20_60"] = f["vol_20d"] / f["vol_60d"].replace(0, np.nan)
    f["skew_60d"] = r1.rolling(60).skew()
    f["dist_60d_high"] = c / h.rolling(60).max() - 1
    f["dist_60d_low"] = c / l.rolling(60).min() - 1
    f["up_days_20"] = (r1 > 0).astype(float).rolling(20).mean()
    f["illiq_20d"] = (r1.abs() / (c * v).replace(0, np.nan)).rolling(20).mean()     # Amihud, used via ranks
    to = (c * v).rolling(20).median()
    to60 = (c * v).rolling(60).median()
    f["turnover_trend"] = np.log((to / to60).where((to > 0) & (to60 > 0)))        # NaN, not -inf, on zero-volume stretches
    if benchmark is not None:
        b0 = benchmark.reindex(df.index).ffill()
        br = b0.pct_change()
        f["beta_60d"] = r1.rolling(60).cov(br) / br.rolling(60).var()
        f["resid_mom_60d"] = c.pct_change(60) - f["beta_60d"] * b0.pct_change(60)
    else:
        f["beta_60d"] = np.nan
        f["resid_mom_60d"] = np.nan

    if benchmark is not None:
        b = benchmark.reindex(df.index).ffill()
        f["rel_strength_20d"] = c.pct_change(20) - b.pct_change(20)
        f["rel_strength_60d"] = c.pct_change(60) - b.pct_change(60)
    else:
        f["rel_strength_20d"] = np.nan
        f["rel_strength_60d"] = np.nan

    return f.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = list(build_features(pd.DataFrame(
    {"open": [1.0] * 5, "high": [1.0] * 5, "low": [1.0] * 5, "close": [1.0] * 5, "volume": [1.0] * 5},
    index=pd.date_range("2020-01-01", periods=5))).columns)


def _nz(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def rule_score(row: pd.Series) -> float:
    """Transparent rule-based score 0-100 of the classic trend/breakout checklist (a transparent baseline to compare the ML model)."""
    g = lambda k: _nz(row.get(k))
    pts = 25 * g("trend_ok") + 10 * g("higher_highs") + 10 * g("higher_lows") + 10 * g("breakout_vol")
    pts += 10 * g("pullback_entry") + 10 * g("cross_50_200") + 5 * g("consolidation_vol") + 10 * g("macd_above_signal")
    pts += 10 * float(50 <= g("rsi14") <= 70) - 20 * g("breakdown_20")
    return float(np.clip(pts, 0, 100))
