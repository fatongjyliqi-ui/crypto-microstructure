# crypto-microstructure
Market microstructure of BTCUSDT (Binance spot + USD-M perp) over 1–2 days:
quoted/effective spread, markouts, spot-perp basis, funding, and a comparison
of Avellaneda-Stoikov optimal quotes with observed spreads. Built for a
Keyrock interview; I must be able to explain every line.

## Rules
- Same layout as my repo portfolio-analytics: src/, tests/, pyproject.toml, GitHub Actions.
- Data -> Parquet in data/ (gitignored). Polars, lazy where possible.
  - data.binance.vision: perp bookTicker, aggTrades, fundingRate (monthly); spot trades.
  - Tardis.dev free tier (1st of each month, no API key): spot quotes. These are 100 ms
    snapshots from depth@100ms; `timestamp` = exchange time, `local_timestamp` = Tardis receipt.
  - Days: 2024-03-01 is the main day (spot + perp). 2024-03-12 is a second, perp-only day
    (US CPI 12:30 UTC). Perp bookTicker history ends 2024-03-30.
  - The live recorder (record.py) is an optional tool; the analysis does not depend on it.
- I write the metric functions myself (effective spread, markouts, AS model). Explain and review them, don't write them unless I ask.
- Every metric gets a test on a small synthetic order book with hand-computed values.
- English for code, comments, README. Run pytest before every commit.
