# crypto-microstructure
Market microstructure of BTCUSDT (Binance spot + USD-M perp) over 1–2 days:
quoted/effective spread, markouts, spot-perp basis, funding, and a comparison
of Avellaneda-Stoikov optimal quotes with observed spreads. Built for a
Keyrock interview; I must be able to explain every line.

## Rules
- Same layout as my repo portfolio-analytics: src/, tests/, pyproject.toml, GitHub Actions.
- Data: data.binance.vision daily files -> Parquet in data/ (gitignored). Polars, lazy where possible.
- I write the metric functions myself (effective spread, markouts, AS model). Explain and review them, don't write them unless I ask.
- Every metric gets a test on a small synthetic order book with hand-computed values.
- English for code, comments, README. Run pytest before every commit.
