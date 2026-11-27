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
The supplied FX file is not yet automatically fetched from NBP.
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
