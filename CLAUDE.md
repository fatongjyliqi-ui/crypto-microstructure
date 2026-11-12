# crypto-microstructure
Market microstructure of Binance USD-M perps (BTCUSDT plus six other USDT perps) over two
days: quoted/effective spread, maker markouts by depth, fill intensity, and the exact
Guéant-Lehalle-Fernandez-Tapia optimal spread compared with observed and break-even
spreads. Spot data and funding rates are downloaded, but the spot-perp basis and funding
are not analysed yet. Portfolio project on market microstructure and market making; I must
be able to explain every line.

## Rules
- Same layout as my repo portfolio-analytics: src/, tests/, pyproject.toml, GitHub Actions.
- Data -> Parquet in data/ (gitignored). Polars, lazy where possible.
  - data.binance.vision: perp bookTicker, aggTrades, fundingRate (monthly); spot trades.
  - Tardis.dev free tier (1st of each month, no API key): spot quotes. These are 100 ms
    snapshots from depth@100ms; `timestamp` = exchange time, `local_timestamp` = Tardis receipt.
  - Days: 2024-03-01 is the main day (seven perps, plus BTCUSDT spot). 2024-03-12 is a
    second, perp-only day for BTCUSDT and XRPUSDT (US CPI 12:30 UTC). Perp bookTicker
    history ends 2024-03-30.
  - The live recorder (record.py) is an optional tool; the analysis does not depend on it.
- I write the metric functions myself (effective spread, markouts, AS model). Explain and review them, don't write them unless I ask.
- Every metric gets a test on a small synthetic order book with hand-computed values.
- English for code, comments, README. Run pytest before every commit.
