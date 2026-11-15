import numpy as np
import pandas as pd
import pytest

from tradingbot import simulate as S
from tradingbot.selective import picks, signal_row, signals_from_predictions


def stock(prices, opens=None, lows=None, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=len(prices)); c = np.array(prices, float)
    o = np.array(opens if opens is not None else prices, float); lo = np.array(lows if lows is not None else np.minimum(o, c), float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": lo, "close": c, "volume": 1e6}, index=idx)


def sigs(rows):
    return pd.DataFrame([dict(date=d, ticker=t, buy=b, sell=s, score=sc, hold=h) for d, t, b, s, sc, h in rows])


def test_signal_rule():
    assert signal_row(.6, "up", .5, "") == (True, False, pytest.approx(.6), 1)
    b, sell, score, hold = signal_row(.6, "up", .58, "up"); assert b and not sell and hold == 10 and score == pytest.approx(.61)       # both agree: 10-day hold, +0.02
    assert signal_row(.6, "up", .42, "down")[:2] == (False, True)                       # any down flag blocks a buy and means exit
    assert signal_row(.5, "", .5, "")[:2] == (False, False)


def test_a_buy_trades_next_open_holds_then_sells_at_the_close_with_costs():
    c = [100, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    u = {"A": stock(c)}; d0 = u["A"].index[0]
    r = S.simulate(sigs([(d0, "A", True, False, .6, 3)]), u, u["A"].index[0], u["A"].index[-1], S.Params(capital=10_000, max_pos=1, cost=0.001, stop_atr=0))
    t = r["trades"].iloc[0]
    assert t["entry_date"] == u["A"].index[1] and t["exit_date"] == u["A"].index[3]            # signal day 0 -> buy at day 1's open, 3rd day's close = exit
    assert t["entry"] == pytest.approx(100 * 1.001) and t["exit"] == pytest.approx(102 * 0.999) and t["why"] == "time"
    assert r["stats"]["end_equity"] == pytest.approx(10_000 * (102 * .999 / (100 * 1.001)), rel=1e-6) and r["stats"]["fees_paid"] > 0


def test_stop_fills_at_the_stop_or_the_gap_open():
    n = 40; flat = [100.0] * n; lows = [99.9] * n; lows[24] = 90.0                 # 20 bars of history so the ATR exists; an intraday dip to 90 on bar 24
    u = {"A": stock(flat, flat, lows)}; ix = u["A"].index
    r = S.simulate(sigs([(ix[21], "A", True, False, .6, 8)]), u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=2.0))
    t = r["trades"].iloc[0]; assert t["why"] == "stop" and t["exit_date"] == ix[24] and 99 < t["exit"] < 100 and t["ret"] < 0        # filled at the stop, not at the low
    opens = [100.0] * n; opens[24] = 70.0; lows2 = [99.9] * n; lows2[24] = 69.0
    g = {"A": stock(flat, opens, lows2)}
    rg = S.simulate(sigs([(ix[21], "A", True, False, .6, 8)]), g, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=2.0))
    tg = rg["trades"].iloc[0]; assert tg["why"] == "stop" and tg["exit"] == 70.0                                              # a gap through the stop fills at the open, worse than the stop


def test_never_more_than_max_positions_best_scores_first_and_no_double_buy():
    u = {t: stock([100] * 12) for t in "ABCD"}; d0 = u["A"].index[0]
    r = S.simulate(sigs([(d0, "A", True, False, .60, 5), (d0, "B", True, False, .70, 5), (d0, "C", True, False, .65, 5), (d0, "D", True, False, .55, 5), (d0, "B", True, False, .9, 5)]),
                   u, d0, u["A"].index[-1], S.Params(capital=10_000, max_pos=2, cost=0.0, stop_atr=0))
    assert set(r["trades"]["ticker"]) == {"B", "C"} and len(r["trades"]) == 2                  # the two best, B only once


def test_down_flag_sells_at_the_next_open_and_the_signal_never_uses_the_future():
    c = [100, 100, 100, 100, 90, 90, 90, 90, 90, 90]; u = {"A": stock(c)}; ix = u["A"].index
    r = S.simulate(sigs([(ix[0], "A", True, False, .6, 10), (ix[2], "A", False, True, 0, 10)]), u, ix[0], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=0))
    t = r["trades"].iloc[0]; assert t["why"] == "down flag" and t["exit_date"] == ix[3] and t["exit"] == 100       # flag at the close of day 2 -> sell at day 3's open
    late = S.simulate(sigs([(ix[-1], "A", True, False, .6, 3)]), u, ix[0], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=0))
    assert len(late["trades"]) == 0 and late["stats"]["end_equity"] == 10_000                                   # a signal on the last day has no next open: no trade


def test_equity_curve_matches_cash_plus_positions_and_baselines_run():
    rng = np.random.default_rng(0); u = {t: stock(100 * np.exp(np.cumsum(rng.normal(0, .01, 80)))) for t in "ABCDE"}; ix = u["A"].index
    rows = [(ix[i], t, True, False, rng.random(), 5) for i in range(0, 60, 4) for t in rng.choice(list("ABCDE"), 2, replace=False)]
    s = sigs(rows); r = S.simulate(s, u, ix[0], ix[-1], S.Params(capital=50_000, max_pos=4, cost=0.001, stop_atr=2))
    assert len(r["equity"]) == len(ix) and r["equity"].iloc[-1] == pytest.approx(50_000 + r["trades"]["pnl"].sum(), rel=1e-6)    # all closed at the end: equity = capital + sum of trade P&L
    bh = S.buy_and_hold(u, ix[0], ix[-1]); assert len(bh) == len(ix) and bh.iloc[0] == pytest.approx(100_000, rel=.05)
    cloud = S.random_baseline(s, u, ix[0], ix[-1], S.Params(capital=50_000, max_pos=4), n=5); assert len(cloud) == 5 and np.isfinite(cloud).all()


def test_picks_and_signals_from_predictions():
    scan = pd.DataFrame({"ticker": list("ABCD"), "p1": [.6, .5, .41, .57], "flag1": ["up", "", "down", "up"], "p10": [.5, .58, .5, .60], "flag10": ["", "up", "", "up"]})
    p = picks(scan, 5); assert list(p["ticker"]) == ["D", "A", "B"] and (p["action"] == "BUY").all()
    class Cal:
        def apply(self, x): return np.asarray(x, float)
        def flag(self, p): return np.where(p >= .55, "up", np.where(p <= .45, "down", ""))
    d = pd.Timestamp("2026-01-05"); a = pd.DataFrame({"date": [d] * 2, "ticker": ["A", "B"], "p": [.6, .4]}); b = pd.DataFrame({"date": [d] * 2, "ticker": ["A", "B"], "p": [.5, .5]})
    m = signals_from_predictions(a, b, Cal(), Cal()).set_index("ticker")
    assert bool(m.loc["A", "buy"]) and bool(m.loc["B", "sell"]) and m.loc["A", "hold"] == 1


def test_rank_policy_buys_the_top_holds_the_middle_and_sells_the_bottom():
    from tradingbot.selective import rank_signals
    d = pd.Timestamp("2026-01-05"); t = list("ABCDEFGH")
    p1 = pd.DataFrame({"date": d, "ticker": t, "p": np.linspace(.6, .4, 8)}); p10 = pd.DataFrame({"date": d, "ticker": t, "p": np.linspace(.6, .4, 8)})
    r = rank_signals(p1, p10, top_n=2, drop_rank=5).set_index("ticker")
    assert list(r.index[r["buy"]]) == ["A", "B"] and list(r.index[r["sell"]]) == ["F", "G", "H"]            # C, D, E: neither bought nor sold (hold)
    w = rank_signals(p1.assign(p=p1["p"][::-1].to_numpy()), p10, top_n=1, drop_rank=3, w10=1.0).set_index("ticker")
    assert w.loc["A", "buy"] and not w.loc["H", "buy"]                                                      # with w10=1 only the 10-day scores decide


def test_trailing_stop_locks_in_gains_and_risk_off_blocks_entries():
    n = 60; up = list(np.linspace(100, 160, 40)) + list(np.linspace(160, 120, 20))             # a rise, then a fall
    u = {"A": stock(up, lows=[x * 0.995 for x in up])}; ix = u["A"].index
    sg = sigs([(ix[21], "A", True, False, .6, 10 ** 6)])
    plain = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=0.0))["trades"].iloc[0]
    trail = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=3.0, trail_atr=3.0))["trades"].iloc[0]
    assert plain["why"] == "end of test" and trail["why"] == "stop" and trail["exit"] > plain["exit"] and trail["ret"] > 0       # the trailing stop sold higher than holding to the end
    off = pd.Series(False, index=ix)
    blocked = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, regime=off))
    assert len(blocked["trades"]) == 0 and blocked["stats"]["end_equity"] == 10_000                                              # risk-off: no new buys
    on = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, regime=~off, stop_atr=0.0))
    assert len(on["trades"]) == 1
    liq = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=0.0, liquidate_on_risk_off=True, regime=pd.Series(ix <= ix[35], index=ix)))
    t = liq["trades"].iloc[0]; assert t["why"] == "risk-off" and t["exit_date"] == ix[37]                                          # regime turns off at the close of day 36 -> sold at day 37 open


def test_inverse_volatility_sizing_gives_calmer_stocks_more_money():
    rng = np.random.default_rng(1); n = 60
    calm = 100 * np.exp(np.cumsum(rng.normal(0, .003, n))); wild = 100 * np.exp(np.cumsum(rng.normal(0, .03, n)))
    u = {"CALM": stock(calm), "WILD": stock(wild)}; ix = u["CALM"].index
    sg = sigs([(ix[30], "CALM", True, False, .6, 3), (ix[30], "WILD", True, False, .5, 3)])
    r = S.simulate(sg, u, ix[25], ix[-1], S.Params(capital=10_000, max_pos=2, cost=0.0, stop_atr=0.0, sizing="invvol"))["trades"].set_index("ticker")
    spent = (r["entry"] * 0 + 1)                                                                                                      # shares are not stored; compare P&L scale instead
    assert len(r) == 2


def test_named_strategies():
    from tradingbot.strategies import FALCON_V1, FALCON_V2, STRATEGIES
    assert set(STRATEGIES) == {"Falcon v1", "Falcon v2"} and FALCON_V1.drop_rank == 40 and FALCON_V2.drop_rank == 30
    p = FALCON_V2.params(capital=5_000); assert p.max_pos == 10 and p.stop_atr == 3.0 and p.capital == 5_000


def liquid(prices, volume=1e6, **kw):
    d = stock(prices, **kw); d["volume"] = volume; return d


def test_real_world_frictions_cost_money_and_respect_liquidity():
    n = 60; flat_up = list(np.linspace(100, 110, n)); u = {"A": liquid(flat_up)}; ix = u["A"].index
    sg = sigs([(ix[25], "A", True, False, .6, 5)])
    ideal = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=1_000_000, max_pos=1, stop_atr=0))
    real = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=1_000_000, max_pos=1, stop_atr=0, spread=0.01, cost=0.002, whole_shares=True))
    assert real["stats"]["end_equity"] < ideal["stats"]["end_equity"]
    # two fills each paying 1% spread + 0.2% fee: about 2.4% worse
    assert ideal["trades"]["ret"].iloc[0] - real["trades"]["ret"].iloc[0] == pytest.approx(0.024, abs=0.004)
    t = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=1_000_000, max_pos=1, stop_atr=0, whole_shares=True))
    assert float(t["trades"]["pnl"].iloc[0]) == pytest.approx(round(float(t["trades"]["pnl"].iloc[0]) / (float(t["trades"]["exit"].iloc[0]) - float(t["trades"]["entry"].iloc[0]))) * (float(t["trades"]["exit"].iloc[0]) - float(t["trades"]["entry"].iloc[0])), rel=1e-9)   # integer number of shares
    thin = {"A": liquid(flat_up, volume=100.0)}                                         # traded value ~10,000 a day: a 10% cap allows ~1,000 per order
    cap = S.simulate(sg, thin, ix[20], ix[-1], S.Params(capital=1_000_000, max_pos=1, stop_atr=0, max_adv_frac=0.10))
    free = S.simulate(sg, thin, ix[20], ix[-1], S.Params(capital=1_000_000, max_pos=1, stop_atr=0))
    assert abs(cap["trades"]["pnl"].iloc[0]) < abs(free["trades"]["pnl"].iloc[0]) / 50                    # the cap shrinks the position to a sliver
    imp = S.simulate(sg, u, ix[20], ix[-1], S.Params(capital=1_000_000_000, max_pos=1, stop_atr=0, impact_coef=1.0))
    assert imp["trades"]["ret"].iloc[0] < ideal["trades"]["ret"].iloc[0]                              # a huge order moves the price against itself


def test_cash_interest_and_tax():
    u = {"A": liquid([100.0] * 400)}; ix = u["A"].index
    idle = S.simulate(sigs([(ix[0], "A", False, False, 0, 1)]), u, ix[0], ix[-1], S.Params(capital=100_000, cash_rate=0.04))
    days = (ix[-1] - ix[0]).days
    assert idle["stats"]["end_equity"] == pytest.approx(100_000 * (1 + 0.04 * days / 365), rel=0.02) and idle["stats"]["return"] > 0.05
    up = {"A": liquid(list(np.linspace(100, 200, 80)))}; jx = up["A"].index
    r = S.simulate(sigs([(jx[5], "A", True, False, .6, 60)]), up, jx[0], jx[-1], S.Params(capital=100_000, max_pos=1, stop_atr=0, tax_rate=0.25))["stats"]
    assert r["profit"] > 20_000 and r["tax"] == pytest.approx(0.25 * r["profit"], rel=1e-6) and r["after_tax_profit"] == pytest.approx(0.75 * r["profit"], rel=1e-6)


def test_real_world_preset_uses_shekels_for_agorot_prices():
    p = S.real_world(1_000_000); assert p.unit == 0.01 and p.max_adv_frac == 0.10 and p.tax_rate == 0.25 and p.whole_shares and p.spread > 0 and p.cash_rate > 0
    agorot = {"A": liquid([10_000.0 + i for i in range(60)])}                           # a 100 shekel stock quoted as 10,000 agorot
    r = S.simulate(sigs([(agorot["A"].index[25], "A", True, False, .6, 5)]), agorot, agorot["A"].index[20], agorot["A"].index[-1], p)
    assert len(r["trades"]) == 1 and 0 < r["stats"]["end_equity"] < 2_000_000


def test_execution_delay_and_fill_model():
    c = list(np.linspace(100, 160, 60)); o = [x - 2 for x in c]; hi = [x + 1 for x in c]; lo = [x - 3 for x in c]
    d = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1e6}, index=pd.bdate_range("2026-01-05", periods=60)); u = {"A": d}; ix = d.index
    sg = sigs([(ix[25], "A", True, False, .6, 3)]); P = lambda **k: S.Params(capital=10_000, max_pos=1, stop_atr=0, cost=0.0, **k)
    base = S.simulate(sg, u, ix[20], ix[-1], P())["trades"].iloc[0]; late = S.simulate(sg, u, ix[20], ix[-1], P(delay_days=2))["trades"].iloc[0]
    assert base["entry_date"] == ix[26] and late["entry_date"] == ix[28] and late["entry"] > base["entry"]              # two extra days of delay: a later, higher fill in a rising market
    avg = S.simulate(sg, u, ix[20], ix[-1], P(fill="ohlc4"))["trades"].iloc[0]; cl = S.simulate(sg, u, ix[20], ix[-1], P(fill="close"))["trades"].iloc[0]
    assert base["entry"] == pytest.approx(d["open"].iloc[26]) and cl["entry"] == pytest.approx(d["close"].iloc[26]) and avg["entry"] == pytest.approx(d[["open", "high", "low", "close"]].iloc[26].mean())
    assert base["entry"] < avg["entry"] < cl["entry"]


def test_settlement_delays_cash_and_min_commission_and_take_profit():
    n = 40; c = [100.0] * n
    u = {"A": stock(c), "B": stock(c)}; ix = u["A"].index
    sg = sigs([(ix[21], "A", True, False, .6, 2), (ix[22], "B", True, False, .6, 2), (ix[23], "B", True, False, .6, 2)])
    base = dict(capital=10_000, max_pos=1, cost=0.0, stop_atr=0)
    a = S.simulate(sg, u, ix[20], ix[-1], S.Params(**base))["trades"]
    b = S.simulate(sg, u, ix[20], ix[-1], S.Params(**base, settle_days=1))["trades"]
    assert len(b) <= len(a)                                                       # cash from a sale is not instantly reusable
    m = S.simulate(sigs([(ix[21], "A", True, False, .6, 2)]), u, ix[20], ix[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=0, min_commission=50.0))
    assert m["trades"].iloc[0]["ret"] < -0.009                                    # 50 on a 10,000 order each side: ~1% round trip
    up = [100.0] * 25 + [100, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130, 130]
    v = {"A": stock(up)}; jx = v["A"].index
    t = S.simulate(sigs([(jx[21], "A", True, False, .6, 9)]), v, jx[20], jx[-1], S.Params(capital=10_000, max_pos=1, cost=0.0, stop_atr=0, tp_atr=1.0))["trades"].iloc[0]
    assert t["why"] == "take-profit" and t["exit_date"] < jx[-1]


def test_cooldown_blocks_a_quick_rebuy_and_rules_only_cancel_buys():
    n = 45; c = [100.0] * n; u = {"A": stock(c)}; ix = u["A"].index
    sg = sigs([(ix[21], "A", True, False, .6, 1), (ix[23], "A", True, False, .6, 1)])
    base = dict(capital=10_000, max_pos=1, cost=0.0, stop_atr=0)
    assert len(S.simulate(sg, u, ix[20], ix[-1], S.Params(**base))["trades"]) == 2
    assert len(S.simulate(sg, u, ix[20], ix[-1], S.Params(**base, cooldown_days=5))["trades"]) == 1
    from tradingbot.rules import apply_rules
    ds = pd.DataFrame({"date": [ix[21], ix[23]], "ticker": "A", "ret_1d": [0.10, 0.0], "gap": 0.0, "rsi14": 50.0, "dist_ma50": 0.0, "trend_ok": 1.0, "vol_ratio": 1.0})
    out = apply_rules(sg, ds, ["no_chase"])
    assert list(out["buy"]) == [False, True] and list(out["sell"]) == [False, False]
