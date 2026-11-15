"""Alpha158-style features (the open-source Qlib feature family used with LightGBM in the literature): candle shape ratios plus, for windows 5/10/20/30/60,
rate of change, moving-average ratio, volatility, rolling-regression slope / R2 / residual, rolling max / min / quantiles / rank, position in the range (RSV),
time since the high / low, price-volume correlation, counts and sums of up / down moves, and volume statistics. All ratios are scaled by the close, so they are
comparable across stocks. Point-in-time: every value at t uses bars <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12
KEYS = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2", "OPEN0", "HIGH0", "LOW0"]
PER_W = ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD",
         "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]
ALPHA_COLUMNS = [f"a_{k}" for k in KEYS] + [f"a_{k}{w}" for k in PER_W for w in WINDOWS]


def alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    f = {}
    hl = h - l + EPS
    f["a_KMID"] = (c - o) / o; f["a_KLEN"] = (h - l) / o; f["a_KMID2"] = (c - o) / hl
    top, bot = np.maximum(o, c), np.minimum(o, c)
    f["a_KUP"] = (h - top) / o; f["a_KUP2"] = (h - top) / hl; f["a_KLOW"] = (bot - l) / o; f["a_KLOW2"] = (bot - l) / hl
    f["a_KSFT"] = (2 * c - h - l) / o; f["a_KSFT2"] = (2 * c - h - l) / hl
    f["a_OPEN0"] = o / c; f["a_HIGH0"] = h / c; f["a_LOW0"] = l / c
    dc = c.diff(); up, dn = dc.clip(lower=0), (-dc).clip(lower=0)
    lv = np.log1p(v); dv = v.diff(); vup, vdn = dv.clip(lower=0), (-dv).clip(lower=0)
    t = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    absr = (c / c.shift(1) - 1).abs() * v
    lvr = np.log1p(v / v.shift(1).replace(0, np.nan))
    cr = c / c.shift(1)
    for w in WINDOWS:
        f[f"a_ROC{w}"] = c.shift(w) / c
        ma = c.rolling(w).mean(); sd = c.rolling(w).std()
        f[f"a_MA{w}"] = ma / c; f[f"a_STD{w}"] = sd / c
        corr_t = c.rolling(w).corr(t)
        slope = corr_t * sd / t.rolling(w).std()
        f[f"a_BETA{w}"] = slope / c; f[f"a_RSQR{w}"] = corr_t ** 2
        f[f"a_RESI{w}"] = (c - (ma + slope * (w - 1) / 2)) / c
        mx, mn = h.rolling(w).max(), l.rolling(w).min()
        f[f"a_MAX{w}"] = mx / c; f[f"a_MIN{w}"] = mn / c
        f[f"a_QTLU{w}"] = c.rolling(w).quantile(0.8) / c; f[f"a_QTLD{w}"] = c.rolling(w).quantile(0.2) / c
        f[f"a_RANK{w}"] = c.rolling(w).rank(pct=True)
        f[f"a_RSV{w}"] = (c - mn) / (mx - mn + EPS)
        imax = h.rolling(w).apply(lambda a: (w - 1 - np.argmax(a)) / w, raw=True)
        imin = l.rolling(w).apply(lambda a: (w - 1 - np.argmin(a)) / w, raw=True)
        f[f"a_IMAX{w}"] = imax; f[f"a_IMIN{w}"] = imin; f[f"a_IMXD{w}"] = imax - imin
        f[f"a_CORR{w}"] = c.rolling(w).corr(lv); f[f"a_CORD{w}"] = cr.rolling(w).corr(lvr)
        cntp = (dc > 0).astype(float).where(dc.notna()).rolling(w).mean(); cntn = (dc < 0).astype(float).where(dc.notna()).rolling(w).mean()
        f[f"a_CNTP{w}"] = cntp; f[f"a_CNTN{w}"] = cntn; f[f"a_CNTD{w}"] = cntp - cntn
        su, sn = up.rolling(w).sum(), dn.rolling(w).sum()
        f[f"a_SUMP{w}"] = su / (su + sn + EPS); f[f"a_SUMN{w}"] = sn / (su + sn + EPS); f[f"a_SUMD{w}"] = (su - sn) / (su + sn + EPS)
        f[f"a_VMA{w}"] = v.rolling(w).mean() / (v + EPS); f[f"a_VSTD{w}"] = v.rolling(w).std() / (v + EPS)
        f[f"a_WVMA{w}"] = absr.rolling(w).std() / (absr.rolling(w).mean() + EPS)
        vu, vn = vup.rolling(w).sum(), vdn.rolling(w).sum()
        f[f"a_VSUMP{w}"] = vu / (vu + vn + EPS); f[f"a_VSUMN{w}"] = vn / (vu + vn + EPS); f[f"a_VSUMD{w}"] = (vu - vn) / (vu + vn + EPS)
    out = pd.DataFrame(f, index=df.index)[ALPHA_COLUMNS].replace([np.inf, -np.inf], np.nan)
    out.index.name = "date"
    return out


def add_alpha(ds: pd.DataFrame, universe: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for tk in ds["ticker"].unique():
        if tk in universe and len(universe[tk]) > 80:
            a = alpha_features(universe[tk]).reset_index(); a["ticker"] = tk; parts.append(a)
    return ds.merge(pd.concat(parts), on=["date", "ticker"], how="left")
