# Data sources and what may be redistributed

This repository contains **code**, and (in `dataset/`) **the project's own outputs**. It does **not** redistribute raw market data.

| Used for | Source | Redistributed here? |
|---|---|---|
| Daily prices, US cross-listings, sector ETFs, macro series, 5-minute bars | Yahoo Finance through `yfinance` | **No.** Downloaded on your machine under Yahoo's terms (personal use). Rebuild with the CLI. |
| Quarterly financial statements | stockanalysis.com (public pages) | **No.** Fetched with `python -m tradingbot fundhist`; respect the site's terms and rate limits. |
| Sector labels, the 125-stock list, dual-listed pairs | Yahoo sector field; TASE index membership; our own research | Yes, small factual tables in `dataset/universe/`. |
| Model scores, simulated trades, equity curves, experiment tables | Produced by this code | Yes, in `dataset/`. Trade logs contain entry/exit prices of individual trades; no price series. |
| Professional-fund 2025 figures in the chart | Press reports (Globes, 20 Jan 2026) | Quoted as reported; gross, not like-for-like. |

If you use this code, you are responsible for complying with each data provider's terms. If you are a rights holder and want something removed, open an issue.
