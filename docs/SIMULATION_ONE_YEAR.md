# One-year trading simulation (paper money, no broker)

Run: 2026-11-12. Code: `scripts/demo_year_*.py` (exactly what produced the numbers; they read `/tmp` files written by the first script), `tradingbot/selective.py`, `tradingbot/simulate.py`.

## What the model does
Every day it ranks the 117 stocks (the improved ensembles: next-day and 10-day questions, blended 70% / 30%). It BUYS a stock that enters the top 20, HOLDS it while it stays in the top 40,
SELLS when it falls below rank 40, or when a 2-ATR stop is hit. Up to 20 equal positions, long only, trades at the next open, 0.1% cost per fill. It is retrained every 21 trading days
on all labels known at that time (walk-forward), so it "learns" from the outcomes of its earlier predictions. The settings (20 positions, rank 40, weight 0.7, stop 2 ATR) were chosen on
2024 only; the two test periods were run once with them. (Earlier feature and model choices were made with the 2025-26 test in view, so those periods are not perfectly untouched.)

## Results (start 100,000)
| Period | Model | Equal-weight buy & hold | TA-125 index | 20-day momentum rank (no AI) | Model delayed 1 extra day |
|---|---|---|---|---|---|
| Calendar 2025 | **+62.9%** (max drawdown -15%, Sharpe 2.8, 653 trades, 60% winners) | +56.7% | +49.0% | +45.4% | +58.1% |
| 2025-11-12 to 2026-11-11 | **+43.9%** (max drawdown -20%, Sharpe 1.7, 784 trades, 56% winners) | +12.8% | +13.5% | +9.6% | +28.0% |

Charts: `docs/demo_2025.png`, `docs/demo_past_year.png`. Fees paid: about 7.8k (2025) and 10.3k (past year) on 100k.

## Read this before believing it
- 2025 was a strong market (index +49%): most of the model's profit is the market; the edge over buying everything is about +6 points. The past year is the more telling test: +31 points over buy & hold.
- Profit is concentrated: 5 stocks gave 31% (2025) and 53% (past year) of it. Luck matters.
- The universe is today's 125 largest stocks: stocks that became large are included, which flatters every long strategy (survivorship bias), including the benchmark.
- Fills are idealised (the open price, 0.1% per side, no market impact) on TASE stocks that are not always liquid.
- A "random picks with the same rules" test loses money (-19%) because constant trading costs money: it is not a fair yardstick, so it is not used. Buy & hold and the momentum rule are.
- The strict "selective" rule (act only on calls whose confidence was validated on earlier data) made zero buys in the past year: the up-calls' confidence bar was not confirmed. The ranking policy above is the active alternative.
- Not tested: quarterly reports and news as inputs. yfinance gives only the last 6-7 quarters of statements, no as-reported dates and only today's news; earnings dates exist for 56 of 117 stocks and mostly only for recent quarters. Backtesting on them would need a paid historical source.

## Names: Falcon v1 (baseline) and Falcon v2 (improved)
`tradingbot/strategies.py`. **Falcon** is our trading model. **Falcon v1** = the baseline above (top 20 / hold inside 40 / 2-ATR stop). **Falcon v2** = improved portfolio settings (below). **Falcon v3** = v2 plus better signals, planned.

### Falcon v2: portfolio-management settings (chosen on 2024 only, 288 settings tried)
Search space: positions (10/20/30), hold-rank multiple (2x/3x), blend weight, stop (2/3 ATR), trailing stop, equal vs volatility-scaled size, market filter (no new buys / sell all below the 100-day average).
Chosen on 2024 (best return with drawdown no worse than 20%): **10 positions, hold inside the top 30, 70% / 30% blend, 3-ATR stop, equal slices, no market filter.**
Then run once on the two test periods:

| Period | Falcon v1 | Falcon v2 | Buy & hold | Index |
|---|---|---|---|---|
| 2024 (selection year) | +58% | +73% | n.a. | n.a. |
| Calendar 2025 | +62.9% | **+93.1%** (drawdown -11%) | +56.7% | +49.0% |
| Past year | +43.9% | **+72.4%** (drawdown -21%) | +12.8% | +13.5% |

Robustness (30 neighbouring settings, `scripts/demo_robustness.py`): every setting beat buy & hold in all three periods, and returns rise steadily as the portfolio gets more concentrated (5 positions best,
30 worst but still +38-47% in the past year). The result is not a lucky spike at one setting.

### Survivorship check (`scripts/demo_point_in_time_check.py`)
Restricting each day to stocks that were really in the 125 most liquid at that date (trailing 60-day turnover over all 527 TASE stocks) removes 17-25% of the signal rows and lowers the results:
| | 2024 | 2025 | Past year |
|---|---|---|---|
| Falcon v1, all 125 -> point-in-time | +58% -> +37% | +63% -> +56% | +44% -> +43% |
| Falcon v2, all 125 -> point-in-time | +73% -> +39% | +93% -> +80% | +72% -> +68% |
The edge survives, but survivorship bias is real, mostly in 2024. Delisted stocks (no data) are still missing, so this is a partial correction.

### Bug fixed along the way
`stop_atr=0` used to place the stop at the entry price (stopping nearly every trade) instead of meaning "no stop". The Falcon v1 numbers used a 2-ATR stop and are unaffected; the "no stop" cells of the first search were wrong and are not used.
