"""Paper-trading simulator: replays the selective rule day by day on past data with realistic frictions. No real money, no broker.

Timing (no look-ahead): a signal is known at the CLOSE of day t (it uses data up to t, and the overnight US session of day t, final by the next morning).
The trade happens at the OPEN of the next trading day. Entry at the open; exit at the close after `hold` trading days, at a stop (2 ATR below the entry fill, filled at
the stop, or at the open if the stock gaps below it), or at the next open after a 'down' flag. A cost is charged on every fill (default 0.1% per side). Money: equal slices
of current equity per position, at most `max_pos` positions, long only (selling means closing a position; shorting is not assumed)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class Params:
    capital: float = 100_000.0
    max_pos: int = 10
    cost: float = 0.001                  # per side, on every buy and every sell
    stop_atr: float = 2.0
    trail_atr: float = 0.0               # > 0: the stop ratchets up to (close - trail_atr x ATR) after every close and never moves down
    sizing: str = "equal"                # "equal" slices, or "invvol": calmer stocks get bigger slices (target 2% daily range, clipped to 0.5x-1.5x); "invvol_score" / "score" add conviction scaling
    regime: object = None                # optional pd.Series[bool] by date (decided at that day's close): False = risk-off, no new buys
    liquidate_on_risk_off: bool = False  # in risk-off, also sell everything at the next open
    # --- real-world frictions (all off by default so the idealised runs stay reproducible; use real_world() for the full set)
    unit: float = 1.0                    # price units -> currency (TASE quotes in agorot: 0.01 turns them into shekels, so capital is in shekels)
    spread: float = 0.0                  # half the bid-ask spread, paid on every fill (0.001 = 0.1%)
    impact_coef: float = 0.0             # extra slippage = coef x daily volatility x sqrt(order value / average daily traded value)
    max_adv_frac: float = 0.0            # never trade more than this share of the stock's average daily traded value (0 = no cap)
    whole_shares: bool = False           # buy whole shares only
    cash_rate: float = 0.0               # yearly interest on idle cash, accrued daily
    tax_rate: float = 0.0                # capital-gains tax on each calendar year's net realised profit, reported as an after-tax figure
    fill: str = "open"                   # where orders are filled: "open" (opening auction), "ohlc4" (average price over the day, as if the order worked all day) or "close" (closing auction)
    settle_days: int = 0                 # sale proceeds become usable cash only this many trading days later (TASE settles T+1 = 1)
    min_commission: float = 0.0          # currency: smallest commission per order (small orders pay more than the % rate)
    thin_adv: float = 0.0                # stocks whose average daily traded value (currency) is below this pay thin_mult x the half-spread
    thin_mult: float = 2.0
    tp_atr: float = 0.0                  # > 0: a standing limit sell at (entry + tp_atr x ATR), hit intraday if the high reaches it (a second daily action)
    cooldown_days: int = 0               # after selling a stock, do not buy it again for this many trading days
    delay_days: int = 0                  # extra trading days between the decision (known at a close) and the order: 0 = next morning, 1 = the morning after that, ...


def real_world(capital: float = 1_000_000.0, **kw) -> "Params":
    """Frictions for trading these stocks with real money in Israel (assumptions, stated): 0.05% commission per fill, 0.1% half-spread, square-root market impact,
    at most 10% of a stock's average daily traded value per order, whole shares, 4% a year on idle cash, 25% capital-gains tax (reported after tax).
    Prices are TASE agorot, so capital is in shekels. Override any field."""
    base = dict(capital=capital, cost=0.0005, spread=0.001, impact_coef=0.5, max_adv_frac=0.10, whole_shares=True, cash_rate=0.04, tax_rate=0.25, unit=0.01,
                settle_days=1, min_commission=7.0, thin_adv=5_000_000.0, thin_mult=2.0)
    base.update(kw)
    return Params(**base)


def _prep(universe: dict[str, pd.DataFrame], unit: float = 1.0) -> dict[str, pd.DataFrame]:
    out = {}
    for tk, df in universe.items():
        d = df[["open", "high", "close", "low"]].copy(); d["atr_now"] = ind.atr(df, 14); d["atr"] = d["atr_now"].shift(1)      # atr: known BEFORE the entry day; atr_now: at the close
        d["adv"] = (df["close"] * df["volume"] * unit).rolling(20).median().shift(1)                                     # average daily traded value in currency, known before the day
        for c in ("open", "high", "close", "low", "atr", "atr_now"):
            d[c] = d[c] * unit
        out[tk] = d
    return out


def _exec_price(p: Params, row) -> float:
    """The price an order is filled at on its execution day, before slippage."""
    if p.fill == "close":
        return float(row["close"])
    if p.fill == "ohlc4":
        return float((row["open"] + row["high"] + row["low"] + row["close"]) / 4)
    return float(row["open"])


def _slip(p: Params, value: float, price: float, atr: float, adv: float) -> float:
    """Slippage fraction for an order of `value` currency: half-spread plus square-root impact."""
    sig = (atr / price) if (np.isfinite(atr) and price > 0) else 0.02
    imp = p.impact_coef * sig * np.sqrt(value / adv) if (p.impact_coef > 0 and np.isfinite(adv) and adv > 0) else 0.0
    sp = p.spread * (p.thin_mult if (p.thin_adv > 0 and np.isfinite(adv) and adv < p.thin_adv) else 1.0)
    return sp + imp


def simulate(signals: pd.DataFrame, universe: dict[str, pd.DataFrame], start, end, p: Params = Params()) -> dict:
    """signals: date, ticker, buy, sell, score, hold (one row per stock-day, decided at that day's close)."""
    px = _prep(universe, p.unit)
    cal = sorted({d for df in universe.values() for d in df.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)})
    sig = {d: g for d, g in signals.assign(date=pd.to_datetime(signals["date"])).groupby("date")}
    reg = p.regime.astype(bool) if p.regime is not None else None
    cash, pos, trades, eq = p.capital, {}, [], []
    last_exit = {}                                                                                # ticker -> day index of its last exit (cooldown)
    pending = []                                                                                  # (day index when usable, proceeds): T+1 settlement
    def recv(x):
        nonlocal cash
        if p.settle_days > 0: pending.append((k + p.settle_days, x))
        else: cash += x
    prev_close_equity = p.capital
    for k, d in enumerate(cal):
        prev = cal[k - 1] if k else None
        for item in [q for q in pending if q[0] <= k]:
            cash += item[1]; pending.remove(item)
        if p.cash_rate > 0 and k:
            cash *= 1 + p.cash_rate * (d - prev).days / 365                                     # idle cash earns interest
        dec = cal[k - 1 - p.delay_days] if k - 1 - p.delay_days >= 0 else None             # the day whose close produced the decision being executed today
        g = sig.get(dec) if dec is not None else None
        # 1) exits requested by a 'down' flag at yesterday's close: sell at today's open
        if g is not None:
            for tk in list(g[g["sell"]]["ticker"]):
                if tk in pos and d in px[tk].index:
                    recv(_close(pos, trades, tk, d, _sell_fill(p, pos[tk], px[tk].loc[d], _exec_price(p, px[tk].loc[d])), "down flag", p.cost))
        risk_on = True if reg is None or prev is None else bool(reg.get(prev, True))
        if not risk_on and p.liquidate_on_risk_off:
            for tk in list(pos):
                if d in px[tk].index:
                    recv(_close(pos, trades, tk, d, _sell_fill(p, pos[tk], px[tk].loc[d], _exec_price(p, px[tk].loc[d])), "risk-off", p.cost))
        # 2) entries at today's open from yesterday's buy signals
        if g is not None and risk_on:
            for _, r in g[g["buy"]].sort_values("score", ascending=False).iterrows():
                tk = r["ticker"]
                if tk in pos or len(pos) >= p.max_pos or tk not in px or d not in px[tk].index:
                    continue
                if p.cooldown_days > 0 and k - last_exit.get(tk, -10 ** 6) <= p.cooldown_days:
                    continue
                o = _exec_price(p, px[tk].loc[d]); atr = px[tk].at[d, "atr"]
                alloc = prev_close_equity / p.max_pos
                if p.sizing in ("invvol", "invvol_score") and np.isfinite(atr) and o > 0:
                    alloc *= float(np.clip(0.02 / (atr / o), 0.5, 1.5))
                if "score" in p.sizing:                                                              # conviction: top-ranked buys get up to 1.3x a slice, the weakest 0.7x
                    alloc *= 0.7 + 0.6 * float(np.clip((float(r["score"]) - 0.9) / 0.1, 0.0, 1.0))
                alloc = min(alloc, cash)
                adv = px[tk].at[d, "adv"]
                if p.max_adv_frac > 0 and np.isfinite(adv):
                    alloc = min(alloc, p.max_adv_frac * adv)                                   # liquidity cap: never more than a fraction of a normal day's trading
                if alloc < 1 or not np.isfinite(o) or o <= 0:
                    continue
                c_eff = max(p.cost, p.min_commission / alloc) if p.min_commission > 0 else p.cost
                fill = o * (1 + _slip(p, alloc, o, atr, adv)) * (1 + c_eff)
                sh = np.floor(alloc / fill) if p.whole_shares else alloc / fill
                if sh <= 0:
                    continue
                cash -= sh * fill
                stop = o - p.stop_atr * atr if (p.stop_atr > 0 and np.isfinite(atr)) else 0.0          # stop_atr <= 0 means no stop
                pos[tk] = dict(entry_date=d, entry=fill, shares=sh, stop=stop, tp=(o + p.tp_atr * atr) if (p.tp_atr > 0 and np.isfinite(atr)) else float("inf"), hold=int(r["hold"]), age=0, score=float(r["score"]), mid=o)
        # 3) stops and time exits during / at the close of today
        for tk in list(pos):
            if d not in px[tk].index:
                continue
            s = pos[tk]; s["age"] += 1; row = px[tk].loc[d]
            if row["low"] <= s["stop"]:
                gross = min(float(row["open"]), s["stop"])             # a gap below the stop fills at the open, otherwise at the stop
                recv(_close(pos, trades, tk, d, _sell_fill(p, s, row, gross), "stop", p.cost))
            elif row["high"] >= s["tp"]:
                recv(_close(pos, trades, tk, d, _sell_fill(p, s, row, max(float(row["open"]), s["tp"])), "take-profit", p.cost))
            elif s["age"] >= s["hold"]:
                recv(_close(pos, trades, tk, d, _sell_fill(p, s, row, float(row["close"])), "time", p.cost))
        if p.trail_atr > 0:                                                  # trailing stop: ratchet up after the close, used from tomorrow
            for tk, s in pos.items():
                if d in px[tk].index and np.isfinite(px[tk].at[d, "atr_now"]):
                    s["stop"] = max(s["stop"], float(px[tk].at[d, "close"]) - p.trail_atr * float(px[tk].at[d, "atr_now"]))
        for t in trades[-50:]:
            if t["exit_date"] == d: last_exit[t["ticker"]] = k
        # 4) mark to market at the close
        val = cash + sum(x for _, x in pending) + sum(s["shares"] * float(px[tk].at[d, "close"]) for tk, s in pos.items() if d in px[tk].index)
        eq.append((d, val)); prev_close_equity = val
    for tk in list(pos):                                                          # close what is still open at the final close
        d = cal[-1]; cash += _close(pos, trades, tk, d, _sell_fill(p, pos[tk], px[tk].loc[d], float(px[tk].at[d, "close"])), "end of test", p.cost)
    cash += sum(x for _, x in pending)
    eq = pd.Series(dict(eq)); eq.iloc[-1] = cash
    t = pd.DataFrame(trades)
    return {"equity": eq, "trades": t, "stats": stats(eq, t, p)}


def _sell_fill(p: Params, s: dict, row, gross: float) -> float:
    """Price actually received: the intended price less slippage on the position's value and the per-fill cost."""
    value = s["shares"] * gross
    c_eff = max(p.cost, p.min_commission / value) if (p.min_commission > 0 and value > 0) else p.cost
    return gross * (1 - _slip(p, value, gross, float(row["atr_now"]) if "atr_now" in row else float("nan"), float(row["adv"]) if "adv" in row else float("nan"))) * (1 - c_eff)


def _close(pos, trades, tk, d, fill, why, cost: float = 0.0) -> float:
    s = pos.pop(tk); proceeds = s["shares"] * fill
    fees = s["shares"] * s["entry"] / (1 + cost) * cost + proceeds / (1 - cost) * cost if cost < 1 else 0.0     # both fills' costs, in money
    trades.append(dict(ticker=tk, entry_date=s["entry_date"], exit_date=d, entry=s["entry"], exit=fill, ret=fill / s["entry"] - 1, pnl=proceeds - s["shares"] * s["entry"],
                       hold_days=s["age"], why=why, score=s["score"], fees=fees))
    return proceeds


def stats(eq: pd.Series, t: pd.DataFrame, p: Params) -> dict:
    r = eq.pct_change().dropna(); dd = (eq / eq.cummax() - 1).min()
    tax = 0.0
    if p.tax_rate > 0 and len(t):
        yr = pd.to_datetime(t["exit_date"]).dt.year; tax = float(sum(max(0.0, g) * p.tax_rate for g in t.groupby(yr)["pnl"].sum()))
    return {"after_tax_profit": float(eq.iloc[-1] - p.capital - tax), "after_tax_return": float((eq.iloc[-1] - p.capital - tax) / p.capital), "tax": tax,
            "start_equity": p.capital, "end_equity": float(eq.iloc[-1]), "profit": float(eq.iloc[-1] - p.capital), "return": float(eq.iloc[-1] / p.capital - 1),
            "max_drawdown": float(dd), "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan"),
            "trades": int(len(t)), "win_rate": float((t["ret"] > 0).mean()) if len(t) else float("nan"), "avg_trade": float(t["ret"].mean()) if len(t) else float("nan"),
            "avg_hold_days": float(t["hold_days"].mean()) if len(t) else float("nan"), "fees_paid": float(t["fees"].sum()) if len(t) else 0.0}


def buy_and_hold(universe: dict[str, pd.DataFrame], start, end, capital: float = 100_000.0) -> pd.Series:
    """Equal-weight buy-and-hold of every stock in the universe from the first open to each close (no costs): what doing nothing clever earns."""
    parts = []
    for tk, df in universe.items():
        d = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if len(d) > 5:
            parts.append(d["close"] / float(d["open"].iloc[0]))
    ew = pd.concat(parts, axis=1).ffill().mean(axis=1)
    return capital * ew


def random_baseline(signals: pd.DataFrame, universe: dict[str, pd.DataFrame], start, end, p: Params = Params(), n: int = 200, seed: int = 0) -> np.ndarray:
    """Final returns of n runs where every day's buy signals are replaced by the SAME NUMBER of random stocks with the same hold lengths and no sell signals:
    what the trading rules earn by luck alone. If the real run is not clearly above this cloud, the model added nothing."""
    rng = np.random.default_rng(seed); tickers = sorted(universe); out = []
    sg = signals.assign(date=pd.to_datetime(signals["date"]))
    buys = sg[sg["buy"]]; days = {d: g for d, g in buys.groupby("date")}
    for _ in range(n):
        rows = []
        for d, g in days.items():
            for hold in g["hold"]:
                rows.append(dict(date=d, ticker=tickers[rng.integers(len(tickers))], buy=True, sell=False, score=float(rng.random()), hold=int(hold)))
        sim = simulate(pd.DataFrame(rows), universe, start, end, p) if rows else None
        out.append(sim["stats"]["return"] if sim else 0.0)
    return np.array(out)
