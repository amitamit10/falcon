"""Quarterly company reports with history: income statement, balance sheet, cash flow and valuation ratios for the last ~20 quarters (about 5 years) of each stock,
from the public pages of stockanalysis.com (robots.txt allows /quote/ pages; one request per second, cached on disk, a generic client name, personal research use).
The site gives the period end date but not the day the report came out, so every quarter becomes usable only `lag` days after its period end (60 days; 90 for the
year-end quarter, which companies report later): later than the real publication in almost all cases, so nothing leaks from the future.
Features at day t use only the latest quarter available at t. Scale-free by design (growth, margins, ratios), because companies report in different currencies;
valuation ratios are re-scaled to today's price: pe_now = pe at the quarter end x close_t / close at the quarter end.
Prices of the stock are TASE agorot; none of the features depends on that unit."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(os.environ.get("TRADINGBOT_FUND_CACHE", "data/fundamentals_hist"))
PAGES = {"income": "", "balance": "balance-sheet/", "cashflow": "cash-flow-statement/", "ratios": "ratios/"}
KEEP = {"income": ["revenue", "gp", "opinc", "netinccmn", "epsdil"], "balance": ["assets", "equity", "debt", "cashneq", "liabilities", "currentLiabilities", "assetsc"],
        "cashflow": ["fcf"], "ratios": ["pe", "ps", "pb", "roe", "roic", "debtequity", "currentratio", "evebitda"]}
FUND_COLUMNS = ["rev_yoy", "rev_qoq", "gp_margin", "op_margin", "net_margin", "ni_yoy", "eps_yoy", "eps_qoq", "fcf_margin", "debt_equity", "cash_assets", "roe_q", "roic_q",
                "pe_now", "ps_now", "pb_now", "earn_yield_now", "days_since_qend", "xs_pe_now", "xs_ps_now", "xs_pb_now", "xs_rev_yoy", "xs_op_margin"]
UA = "tradingbot-research/0.1"


def parse_page(html: str) -> pd.DataFrame | None:
    """The page embeds its table as a JavaScript object `financialData:{datekey:[...],revenue:[...],...}`; turn it into a frame indexed by period end date."""
    i = html.find("financialData:{datekey:"); off = len("financialData:")
    if i < 0:
        i = html.find("data:{datekey:"); off = len("data:")
    if i < 0:
        return None
    j = i + off; depth = 0; k = j
    while k < len(html):
        c = html[k]
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: break
        k += 1
    js = re.sub(r'([{,])([A-Za-z_][A-Za-z0-9_]*):', r'\1"\2":', html[j:k + 1])
    js = re.sub(r'(?<![\d.])(-?)\.(\d)', r'\g<1>0.\2', js).replace("void 0", "null").replace("Infinity", "null").replace("NaN", "null").replace("-null", "null")
    d = json.loads(js)
    n = len(d["datekey"]); df = pd.DataFrame({k: v for k, v in d.items() if isinstance(v, list) and len(v) == n})
    df["datekey"] = pd.to_datetime(df["datekey"], format="%Y-%m-%d", errors="coerce")
    return df.dropna(subset=["datekey"]).set_index("datekey").sort_index()


def _get(url: str) -> str:
    import requests
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_stock(ticker: str, cache: Path | None = None, delay: float = 1.0, refresh: bool = False) -> pd.DataFrame:
    """Quarterly history of one stock (period end -> columns of every page), cached as CSV. ticker like 'TEVA.TA'."""
    cache = Path(cache or CACHE); cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{ticker}.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True)
    base = ticker.replace(".TA", ""); parts = []
    for name, path in PAGES.items():
        try:
            df = parse_page(_get(f"https://stockanalysis.com/quote/tlv/{base}/financials/{path}?p=quarterly"))
        except Exception:  # noqa: BLE001 - a missing page just means no data for that block
            df = None
        time.sleep(delay)
        if df is not None:
            parts.append(df[[c for c in KEEP[name] if c in df.columns]].add_prefix(name[:3] + "_"))
    out = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    out.to_csv(f)
    return out


def fetch_all(tickers: list[str], **kw) -> dict[str, pd.DataFrame]:
    res = {}
    for i, t in enumerate(tickers):
        res[t] = fetch_stock(t, **kw)
    return res


def _avail(idx: pd.DatetimeIndex, lag: int = 60, q4_lag: int = 90) -> pd.DatetimeIndex:
    return idx + pd.to_timedelta([q4_lag if d.month == 12 else lag for d in idx], unit="D")


def _div(a, b):
    return (a / b.where(b.abs() > 0)).replace([np.inf, -np.inf], np.nan)


def quarter_features(h: pd.DataFrame) -> pd.DataFrame:
    """Per-quarter ratios from the raw columns (indexed by period end)."""
    g = lambda c: h[c] if c in h else pd.Series(np.nan, index=h.index)
    f = pd.DataFrame(index=h.index)
    rev, ni, eps = g("inc_revenue"), g("inc_netinccmn"), g("inc_epsdil")
    f["rev_yoy"] = rev / rev.shift(4).where(rev.shift(4) > 0) - 1
    f["rev_qoq"] = rev / rev.shift(1).where(rev.shift(1) > 0) - 1
    f["gp_margin"] = _div(g("inc_gp"), rev); f["op_margin"] = _div(g("inc_opinc"), rev); f["net_margin"] = _div(ni, rev)
    f["ni_yoy"] = _div(ni - ni.shift(4), ni.shift(4).abs()); f["eps_yoy"] = _div(eps - eps.shift(4), eps.shift(4).abs()); f["eps_qoq"] = _div(eps - eps.shift(1), eps.shift(1).abs())
    f["fcf_margin"] = _div(g("cas_fcf"), rev); f["debt_equity"] = _div(g("bal_debt"), g("bal_equity")); f["cash_assets"] = _div(g("bal_cashneq"), g("bal_assets"))
    f["roe_q"], f["roic_q"] = g("rat_roe"), g("rat_roic")
    f["pe_q"], f["ps_q"], f["pb_q"] = g("rat_pe"), g("rat_ps"), g("rat_pb")
    return f


def fund_features(universe: dict[str, pd.DataFrame], hist: dict[str, pd.DataFrame], lag: int = 60, q4_lag: int = 90) -> pd.DataFrame:
    """Long frame date, ticker + FUND_COLUMNS from each stock's daily index, using the latest quarter available (period end + lag) at that day."""
    rows = []
    for tk, px in universe.items():
        h = hist.get(tk)
        base = pd.DataFrame({"date": px.index, "close": px["close"].to_numpy()}); base["ticker"] = tk
        if h is None or len(h) < 2:
            rows.append(base.drop(columns="close")); continue
        q = quarter_features(h.sort_index()); q["qend"] = q.index; q["avail"] = _avail(q.index, lag, q4_lag)
        qclose = px["close"].reindex(q.index, method="ffill").to_numpy(); q["close_q"] = qclose
        base["date"] = pd.to_datetime(base["date"]).astype("datetime64[ns]"); q["avail"] = pd.to_datetime(q["avail"]).astype("datetime64[ns]")      # same time unit on both sides (pandas versions differ)
        m = pd.merge_asof(base.sort_values("date"), q.sort_values("avail"), left_on="date", right_on="avail", direction="backward")
        scale = m["close"] / m["close_q"]
        m["pe_now"] = m["pe_q"] * scale; m["ps_now"] = m["ps_q"] * scale; m["pb_now"] = m["pb_q"] * scale
        m["earn_yield_now"] = np.where(m["pe_now"] > 0, 1 / m["pe_now"], np.nan)
        m["days_since_qend"] = (m["date"] - m["qend"]).dt.days
        rows.append(m[["date", "ticker"] + [c for c in FUND_COLUMNS if c in m.columns and not c.startswith("xs_")]])
    out = pd.concat(rows, ignore_index=True)
    for c in FUND_COLUMNS:
        if c not in out: out[c] = np.nan
    g = out.groupby("date")
    for new, src in (("xs_pe_now", "pe_now"), ("xs_ps_now", "ps_now"), ("xs_pb_now", "pb_now"), ("xs_rev_yoy", "rev_yoy"), ("xs_op_margin", "op_margin")):
        out[new] = g[src].rank(pct=True)
    return out[["date", "ticker"] + FUND_COLUMNS]


def load_cached(tickers) -> dict[str, pd.DataFrame]:
    """Whatever has been downloaded already (no network): stocks without a cache file simply have no report features."""
    out = {}
    for t in tickers:
        f = CACHE / f"{t}.csv"
        if f.exists():
            try:
                out[t] = pd.read_csv(f, index_col=0, parse_dates=True)
            except Exception:  # noqa: BLE001 - a damaged cache file counts as missing
                pass
    return out


def add_fund(ds: pd.DataFrame, universe: dict, hist: dict | None = None, **kw) -> pd.DataFrame:
    """Merge the report features. Reads only the disk cache unless `hist` is given: download once with `python -m tradingbot fundhist`."""
    hist = load_cached(universe) if hist is None else hist
    return ds.merge(fund_features(universe, hist, **kw), on=["date", "ticker"], how="left")
