# crypto-microstructure

Market microstructure of BTCUSDT on Binance spot and the USD-M perpetual: quoted and
effective spread, markouts, spot-perp basis, funding, and a comparison of
Avellaneda-Stoikov optimal quotes with observed spreads.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## Data

Public files from [data.binance.vision](https://data.binance.vision), converted to
Parquet under `data/` (gitignored):

```bash
python -m crypto_microstructure.download --date 2024-03-12
python examples/plot_perp_spread.py
```

| dataset | source path | notes |
|---|---|---|
| `perp_bookTicker` | `futures/um/daily/bookTicker` | only 2023-05-16 to 2024-03-30 |
| `perp_aggTrades` | `futures/um/daily/aggTrades` | |
| `perp_fundingRate` | `futures/um/monthly/fundingRate` | monthly files only |
| `spot_trades` | `spot/daily/trades` | no header; timestamps in µs from 2025 |

There is **no historical spot bookTicker** on data.binance.vision (spot only has
`aggTrades`, `klines`, `trades`), so spot quotes are not available from this source.

### Live quotes

Spot and perp top-of-book are recorded from the Binance websocket `bookTicker` streams
into `data/live/<venue>_bookTicker/<UTC day>/part-*.parquet`:

```bash
caffeinate -i python -m crypto_microstructure.record --hours 48
```

Spot bookTicker messages carry no exchange timestamp, so every row is stamped with
`local_time` (receive time, µs) as the common clock; perp rows also keep the exchange
`transaction_time` / `event_time`. `session` increments on every reconnect.
