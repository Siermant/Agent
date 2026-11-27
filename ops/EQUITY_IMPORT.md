# v0.13 research data import — initial implementation
Offline, no broker access, no orders. Does not change running crypto strategies.
Use independent verified input data with the exact CSV headers below:
- calendar.csv: mic,session,close_at
- bars.csv: instrument_id,ticker,mic,kind,currency,exchange_tz,session,close_at,retrieved_at,raw_close,adjusted_close,adjustment,source
- fx.csv: currency,date,pln_per_unit

Each instrument_id identifies one listing, not merely an issuer or ticker.
Kind stock/etf; currency PLN/EUR/USD. Timestamps include UTC offset; timezone
uses IANA identifier. The calendar must already contain the verified completed
sessions for the requested date window and venue. The importer checks bar
completeness against the supplied calendar, not against an external exchange
calendar. It cannot prove that the supplied calendar, prices or FX are genuine,
that all requested instruments are present, or that a declared adjustment
policy is correct. A newly listed instrument needs its own aligned window.
Use only legitimate data with appropriate rights. Do not store credentials
or licensed raw datasets in a public repository.

Adjustment none (empty adjusted_close), split or split_dividend (separate
positive adjusted_close). Do not treat adjusted_close as an executable quote.
FX uses strictly earlier daily reference rate, at most seven calendar days
old. PLN is identity conversion. Output JSON is research-only; no orders.
FX CSV can now be generated independently from NBP Table A (see below).
Historical prices may be revised retrospectively; point-in-time corporate
action information is not established by this importer.
```sh
python3 -m global_agent.equity_data --calendar calendar.csv --bars bars.csv --fx fx.csv --output equity_research.json
```
This command does not interact with the active service or SQLite ledger.
Test with synthetic fixtures:
```sh
python3 -m unittest discover -s tests -p test_equity_data.py -v
```
Nine local tests passed; source and test Git blob hashes matched the tested
copies. Full suite and Docker image not yet validated for this increment.
Next: provider integration, calendar authority, requested-instrument manifest,
daily import storage and independent freshness state, then paper simulation.

## Provider adapters (development)
NBP reference FX exporter, no credentials:
```sh
python3 -m global_agent.fx_reference --start 2026-11-01 --end 2026-11-26 --output fx.csv
```
Uses EUR/USD Table A, chunks of at most 90 days, opening prior-rate buffer,
atomic publication. Missing rates in the interior are checked against each
session by equity_data. It does not produce executable FX or tax valuations.

US historical Alpaca IEX adapter (not connected to a real account yet):
```sh
python3 -m global_agent.alpaca_data --manifest instruments.csv --calendar calendar.csv --output bars.csv
```
The instrument manifest has exact columns instrument_id,ticker,mic,kind.
Use 1–30 explicitly identified US listings, with a calendar provided and
checked independently by the operator. The calendar must contain prior
completed New York dates; it is NOT automatically verified against holidays.
The adapter requires APCA_API_KEY_ID and APCA_API_SECRET_KEY supplied privately
to the local process. Do not put keys in Git, diagnostic reports or chat.
Only HTTPS GET to data.alpaca.markets, no redirects, no brokerage orders.
IEX is a single-venue feed, not a consolidated market quote. Daily bar
aggregates are not necessarily official closing auction prices; close_at is
the calendar session boundary, not the timestamp of the last trade.
Raw close and split/dividend adjusted close are downloaded separately.
The adapter checks exact requested symbols and dates, OHLC, bounded
pagination, no current day, and rejects missing data before publishing CSV.
No automatic trading, no entitlement checks for IKE or EU retail, no
point-in-time corporate action model. Actual credentials, API entitlements,
data licensing and live responses are not verified yet.
References: https://api.nbp.pl/en.html ; https://docs.alpaca.markets/us/reference/stockbars ;
https://docs.alpaca.markets/us/docs/market-data-faq
