# Falcon v3 run

Chosen on the validation year 2024-01-02 to 2024-12-31 only: recipe **cross**, 20 positions, hold inside the top 60, 3.0-ATR stop.

## Signal recipes on the validation year (higher score is better; score = 0.7 x 10-day rank IC + 0.3 x next-day rank IC)

| Recipe | AUC next day | AUC 10 days | Rank IC next day | Rank IC 10 days | Score | Minutes |
|---|---|---|---|---|---|---|
| cross | 0.5530 | 0.5418 | 0.0876 | 0.1083 | 0.1021 | 4 |
| full | 0.5526 | 0.5418 | 0.0914 | 0.0995 | 0.0971 | 19 |
| big | 0.5579 | 0.5413 | 0.0620 | 0.1034 | 0.0910 | 3 |
| base | 0.5564 | 0.5397 | 0.0540 | 0.1067 | 0.0909 | 5 |
| ens5 | 0.5564 | 0.5418 | 0.0540 | 0.1031 | 0.0884 | 2 |
| fast10 | 0.5510 | 0.5382 | 0.0530 | 0.0916 | 0.0800 | 9 |
| longwin | 0.5370 | 0.5393 | 0.0524 | 0.0884 | 0.0776 | 7 |

Validation year, real-world, best portfolio for the base engine (20/60/3.0): +30%, Sharpe 2.10. Chosen v3: +36%, Sharpe 2.53.

## Professional managers, for scale (calendar 2025)

Published results (Globes, 20 Jan 2026): Israeli *mutual hedge funds* hold 0-100% shares and may use leverage; returns are gross (after management fees, before the ~20% success fee). They are a yardstick, not a like-for-like comparison with a long-only stock strategy. TA-125 index in 2025: +50% (ynet, 29 Dec 2025).

| Fund | 2025 return |
|---|---|
| Harel Multi-Strategy (top mutual hedge fund) | +78% |
| Meitav Long Short (mutual hedge fund) | +68% |
| Average of 46 Israeli mutual hedge funds | +25% |

No published 2025 return table for plain actively managed equity funds could be retrieved (the main fund-data site refused automated access); the Calcalist half-year review said most large equity funds beat the index in the first half of 2025, without figures.

## Calendar 2025

Benchmarks: buy everything and hold +56.7%, TA-125 index +49.0%.

### Ideal

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +65.1% | +65.1% | -15% | 2.86 | 675 | 59% | 8,115 |
| Falcon v2 | +83.4% | +83.4% | -9% | 3.52 | 305 | 66% | 7,954 |
| Falcon v3 | +73.7% | +73.7% | -12% | 3.27 | 340 | 65% | 4,119 |

### Real-world

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +40.4% | +30.4% | -16% | 2.00 | 675 | 56% | 35,370 |
| Falcon v2 | +46.2% | +34.8% | -12% | 2.33 | 305 | 59% | 31,615 |
| Falcon v3 | +56.9% | +42.8% | -13% | 2.74 | 340 | 61% | 18,585 |

### Real-world, order during the day

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +37.8% | +28.4% | -15% | 1.95 | 656 | 55% | 33,967 |
| Falcon v2 | +35.8% | +26.9% | -15% | 1.85 | 304 | 54% | 30,063 |
| Falcon v3 | +57.0% | +42.8% | -13% | 2.73 | 340 | 61% | 18,436 |

### Real-world, one day late (stress test)

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +31.7% | +23.8% | -16% | 1.72 | 641 | 51% | 32,098 |
| Falcon v2 | +28.2% | +21.3% | -15% | 1.52 | 304 | 52% | 29,079 |
| Falcon v3 | +46.6% | +35.1% | -12% | 2.38 | 341 | 57% | 17,985 |


## Past year

Benchmarks: buy everything and hold +12.8%, TA-125 index +13.5%.

### Ideal

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +50.3% | +50.3% | -17% | 1.80 | 787 | 56% | 10,599 |
| Falcon v2 | +73.3% | +73.3% | -19% | 2.21 | 365 | 58% | 11,144 |
| Falcon v3 | +55.1% | +55.1% | -21% | 1.89 | 370 | 62% | 5,123 |

### Real-world

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +25.7% | +18.9% | -25% | 1.06 | 787 | 53% | 47,798 |
| Falcon v2 | +38.8% | +29.2% | -29% | 1.38 | 365 | 53% | 49,125 |
| Falcon v3 | +43.8% | +32.9% | -23% | 1.59 | 370 | 58% | 24,302 |

### Real-world, order during the day

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +17.3% | +12.8% | -28% | 0.79 | 750 | 52% | 43,947 |
| Falcon v2 | +27.6% | +20.7% | -31% | 1.06 | 363 | 49% | 46,760 |
| Falcon v3 | +37.5% | +28.2% | -23% | 1.42 | 380 | 56% | 24,221 |

### Real-world, one day late (stress test)

| Strategy | Return | After tax | Max drawdown | Sharpe | Trades | Win rate | Fees |
|---|---|---|---|---|---|---|---|
| Falcon v1 | +4.9% | +3.4% | -29% | 0.33 | 744 | 45% | 40,834 |
| Falcon v2 | +17.3% | +13.0% | -33% | 0.72 | 351 | 48% | 42,732 |
| Falcon v3 | +30.1% | +22.6% | -23% | 1.20 | 379 | 52% | 23,374 |

## What the scenarios mean (the timeline of one trade)
- The model runs in the morning, before the TASE opens (05:00 UTC), once the previous US session is final: the data of day t, including the US close, is known on the morning of day t+1.
- **Ideal**: the order fills at that day's opening price; only a flat 0.1% cost per trade.
- **Real-world**: the same timing, as an automated order in the opening auction, but with spread, market impact, a liquidity cap, whole shares, interest on idle cash and tax. This is the intended way of running it.
- **Real-world, order during the day**: the order is placed some time after the open (a person reading the report, or a slower connection): filled at that day's average price (mean of open, high, low and close). A reasonable model of a delay of a few hours.
- **Real-world, one day late (stress test)**: a whole extra day is lost (the order fills the day after, at its average price). This is a pessimistic worst case, not a delay to expect.
- Daily bars cannot model delays of minutes exactly; the only intraday history available (60 days of 5-minute bars) is too short to calibrate one.

## How to read this
- Everything is paper money. Prices are TASE agorot; real-world runs are in shekels with 1,000,000 to start.
- 'After tax' applies 25% to each calendar year's net realised profit in the real-world scenarios; the Ideal scenario has no tax, so there it equals the return.
- The settings were chosen on the validation year only; the test periods were run once. Earlier feature and model choices were made with 2025-26 in view, so those periods are not perfectly untouched.
- Survivorship: the universe is today's largest stocks. The point-in-time section (when present) restricts each day to the stocks that were really among the 125 most liquid then.

## Notes added after reading the numbers (2026-11-13)
- **Where v3 wins and where it does not.** In the Ideal scenario v2 is best in both test periods. v3 is best in all the Real-world comparisons (opening auction, order during the day, one day late; both periods): it trades about half as often, so it pays about half the costs, and its signals hold up better when the order is a day late (v1 and v2 fall to +5% and +17% in the past year; v3 keeps +30%).
- **2025 was a strong market.** In the Real-world scenario v3 (+56.9%) only matches buying everything and holding (+56.7%, before tax); after the 25% tax on realised gains (+42.8%) it is behind. The model's value shows in the choppier past year: +43.8% vs +12.8% (about +33% after tax vs buy and hold before tax).
- **Why v3 improved.** The 2024-only selection found one real source of new information: US prices of dual-listed companies (next-day rank IC 0.054 to 0.088 on 2024). More frequent retraining, more history and larger models did not help (longer history hurt), so those were not chosen.
- **Still flattered by:** survivorship (today's largest stocks; the earlier point-in-time check cut Falcon v2's 2024 return roughly in half and left 2025 and the past year about 5-10% lower), a few big winners, the simplified tax and cost models, and earlier feature and model choices that were made with 2025-26 in view. Max drawdown in the past year reached -23% (v3) and -29% (v2) in the Real-world run.
- Only a live, no-money track record over several months can show whether any of this holds.
- **About delay.** The one-day-late case is a stress test (a whole extra day lost), not an expected delay. The realistic delay case is "order during the day" (filled at the day's average price): v1 and v2 lose 8-11 points against the opening-auction run, v3 loses nothing in 2025 and about 6 points in the past year.
- **Professional managers (2025 only).** Published results of Israeli mutual hedge funds (Globes, Jan 2026; gross of ~20% success fee; 0-100% shares, possibly leveraged): Harel Multi-Strategy +78%, Meitav Long Short +68%, average of 46 funds +25%. Falcon v3 (+57% real-world, before tax) is above the average and below the two standouts. A yardstick, not a like-for-like comparison.
