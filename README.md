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

Two days, converted to Parquet under `data/` (gitignored):

- **2024-03-01** is the main day, with spot and perp data.
- **2024-03-12** is a second, perp-only day (US CPI release at 12:30 UTC).

```bash
python -m crypto_microstructure.download --date 2024-03-01
python -m crypto_microstructure.download --date 2024-03-12
python examples/plot_perp_spread.py --date 2024-03-01
python examples/plot_perp_spread.py --date 2024-03-12
```

| dataset | source | notes |
|---|---|---|
| `perp_bookTicker` | data.binance.vision `futures/um/daily/bookTicker` | only 2023-05-16 to 2024-03-30 |
| `perp_aggTrades` | data.binance.vision `futures/um/daily/aggTrades` | |
| `perp_fundingRate` | data.binance.vision `futures/um/monthly/fundingRate` | monthly files only |
| `spot_trades` | data.binance.vision `spot/daily/trades` | no header; timestamps in µs from 2025 |
| `spot_quotes` | [Tardis.dev](https://tardis.dev) `binance/quotes` | free tier: 1st of each month only |

### Spot quotes come from Tardis

data.binance.vision has **no spot bookTicker** (spot only has `aggTrades`, `klines` and
`trades`). Spot best bid/ask therefore comes from the Tardis.dev free tier. Points to
keep in mind:

- **100 ms snapshots, not every change.** Tardis rebuilds the quotes from Binance's
  `depth@100ms` stream. For 2024-03-01, 94% of the gaps between updates are multiples
  of 100 ms. Quote changes within one 100 ms window are not visible, so a trade
  matched to the latest quote can be matched to a quote up to ~100 ms old.
- **`timestamp`** is Binance's exchange event time (ms precision).
  **`local_timestamp`** is when Tardis received it, a median ~4 ms later.
- The perp `bookTicker`, by contrast, records every top-of-book change, with exchange
  times `transaction_time` / `event_time`.

### Known quirk: off-grid perp quotes

On 2024-03-01, 274 of 39M perp bookTicker rows have a price off the 0.1 tick grid
(e.g. an ask of 61431.87 for 0.001 BTC, 04:25–20:30 UTC). 2024-03-12 has none. They
are kept in the Parquet files. Don't take the tick size as the smallest observed
spread: it comes out as 0.06 that day.

### Optional: live quotes

The analysis does not depend on this. `record.py` records the Binance websocket
`bookTicker` streams for spot and perp into `data/live/<venue>_bookTicker/<UTC day>/part-*.parquet`:

```bash
caffeinate -i python -m crypto_microstructure.record --hours 48
```

Spot bookTicker messages carry no exchange timestamp, so every row is stamped with
`local_time`, the receive time in µs, as the common clock. Perp rows also keep the
exchange `transaction_time` / `event_time`. `session` increments on every reconnect.

## Limitations

- **Fees are not included.** Markouts, break-even spreads and model spreads are before
  maker/taker fees and rebates. Binance's current pages list 0.02% maker / 0.05% taker
  for the base USDⓈ-M futures tier, but I found no source dated March 2024, and the fee
  of the makers in the data (VIP tier, rebates, liquidity-provider programs) is unknown.
  At the BTCUSDT perp touch the half-spread is ~0.008 bps, far below any plausible
  non-rebated maker fee, so the touch is only profitable with rebates or a high tier.
