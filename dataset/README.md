# Falcon dataset

The project's own outputs, so you can inspect, re-score or build on them without re-running hours of training. **Paper-trading research, not investment advice.**
No raw price series or financial statements from third parties are included (see [`../DATA_SOURCES.md`](../DATA_SOURCES.md)). Rebuild everything with `scripts/export_dataset.py`.

License of these files: **CC BY 4.0** (attribute the project). The code is MIT.

## `signals/`
Daily model scores, one row per stock-day, walk-forward: the score for day *t* comes from a model trained only on data known at *t* (retrained every 21 days).

| File | Rows | Coverage | Model |
|---|---|---|---|
| `falcon_v4_scores.csv.gz` | 92,146 | 2023-01-03 to 2026-11-11 | recipe `cross2r` (sector/cross-asset flow, US cross-listings, rank models) |
| `falcon_v3_scores.csv.gz` | 47,112 | 2025-01-02 to 2026-11-11 | recipe `cross` (US cross-listing signals) |

Columns: `date`, `ticker` (Yahoo style, `.TA`), `p_next_day_up` (calibrated probability the stock closes up tomorrow), `p_10day_beats_median` (probability that its 10-day return beats the
day's median), `score_blend` (70% rank of the 10-day question + 30% rank of the next-day question, in [0,1]), `rank` (1 = best that day). 117 of the 125 index stocks have enough history to be scored.
The 2023-24 rows were the validation period used to choose settings; 2025-26 are the test periods. A probability is not a promise: the models' AUC is about 0.55-0.57.

## `results/`
Simulated trading with the strict real-world rules (1,000,000 ILS; opening-auction fills, commission with a minimum, spread, market impact, T+1 settlement, whole shares; 25% tax is reported separately in `summary.json`).

| File | Content |
|---|---|
| `summary.json` | per period and model: return, after-tax return, max drawdown, Sharpe, trades, fees; plus index and equal-weight buy&hold returns |
| `<period>_equity.csv` | daily equity (ILS) of v3 and v4, and the TA-125 index and equal-weight buy&hold rebased to 1 |
| `<period>_falcon_v3_trades.csv`, `..._v4_trades.csv` | every simulated trade: ticker, entry/exit date and price (shekels), return, P&L, days held, why it exited, fees |
| `experiments_*.csv` | the validation-period (2024) experiments: realism checks, portfolio search, rule guards, ensembles, recipe selection |

Periods: `calendar_2025`, `2026_so_far` (to 2026-11-11), `past_year` (2025-11-12 to 2026-11-11).

## `universe/`
`tase125.txt` (the 125 stocks: today's list, so survivorship bias applies), `sectors.csv` (Yahoo sector/industry), `dual_listed.csv` (TASE companies also listed in the US, used for the cross-listing signals).

## Caveats
Survivorship bias (today's index members), 2025 was examined in earlier iterations before the final single run, several candidates were compared, and simulated fills are not real fills.
Headline numbers and everything that did not help: [`../docs/FALCON_V4_RESULTS.md`](../docs/FALCON_V4_RESULTS.md).
