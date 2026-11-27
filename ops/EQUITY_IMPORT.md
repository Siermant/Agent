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

## Research ranking (momentum_risk_v1)
Run after the validated import:
```sh
python3 -m global_agent.equity_rank --input equity_research.json
```
61 completed observations, adjusted PLN closes with split/dividend policy.
Separate cohorts for identical session dates, listing MIC and data source.
20- and 60-session returns, annualized sample volatility (252-session
convention) and maximum drawdown in the 61-observation window.
Heuristic score = 0.4 * return20 + 0.6 * return60 - 0.25 * volatility
- 0.5 * max drawdown, all in percentage points. Unoptimized, not a return
forecast. Reject duplicate or future data; exclude series older than five
calendar days, insufficient observations or missing adjusted prices.
Calendar verification and point-in-time corporate actions are still absent.
This is a retrospective screen, not a backtest or trade recommendation.
Liquidity, costs, sector exposure and instrument eligibility not yet
evaluated; they must be added before use for portfolio decisions.

## Hypothetical Global initial allocation
```sh
python3 -m global_agent.equity_rank --input equity_research.json > ranking.json
python3 -m global_agent.global_plan --ranking ranking.json --assumptions assumptions.json --cohort 0
```
One comparable cohort per run, five separate hypothetical accounts of
50,000 PLN, no existing holdings. A-D use different transparent momentum/risk
weights; positive scores only. D is a fixed heuristic, not AI. Position
budget 4.5% of starting capital including estimated entry costs; exposure
targets 40/60/80/80%, minimum 20% cash, proposed 20% per supplied sector label.
E is a separately identified ETF benchmark, exempt from active allocation
limits but subject to assumed liquidity and cash. No share quantities or
executable orders. Assumptions JSON must explicitly supply research_only=true,
commission_bps, spread_slippage_bps, minimum_fee_pln, minimum_adv_pln,
max_adv_fraction, benchmark_id, instruments keyed by listing ID with
research_eligible, sector, adv_pln and optional benchmark_eligible.
No automatic inference of ADV or sector: missing metadata excluded.
Notional amounts are hypothetical, not broker quotes; costs cover entry
only; liquidity and sector labels are unverified, no ETF look-through.
No ledger updates, no historical portfolio simulation or live trading.
Next: validated metadata/volume data and independent calendar source,
then paper fills with executable quotes, existing holdings, and ledger risk
checks. IKE eligibility and EU ETF selection remain separate.

## Integrated research cycle
Offline:
```sh
python3 -m global_agent.research_cycle --calendar calendar.csv --bars bars.csv --fx fx.csv --assumptions assumptions.json --output research.json
```
With a separately authorized Alpaca data credential in the local environment:
```sh
python3 -m global_agent.research_cycle --calendar calendar.csv --alpaca-manifest instruments.csv --assumptions assumptions.json --output research.json
```
The second mode downloads IEX daily bars and NBP reference FX into a temporary
directory, validates the dataset, ranks instruments and builds hypothetical
initial allocations. Raw downloads are removed after the cycle. The output
is research-only and never goes to the live ledger or diagnostic repository.
Use a dedicated research directory for inputs/output, not the service volume.
A single comparable cohort can be selected with --cohort N (zero based).
If there are multiple cohorts and none is selected, publish the ranking only
with state cohort_selection_required and exit code 2; no duplicate capital
allocation across venues. Insufficient history produces no_rankable_instruments
and exit code 2. Input errors exit 1 without publishing a new result;
consumers must check generated_utc and state rather than infer freshness from
file existence. Output is replaced atomically on successful computation.
The provider path still requires real API validation and a verified calendar,
metadata and cost/liquidity assumptions. Tests use synthetic data or mocks,
not actual market data. Not a scheduled process, no execution or deployment.

## Synthetic Global execution laboratory
```sh
python3 -m unittest discover -s tests -p test_global_sim.py -v
python3 -m global_agent.global_sim --scenario scenario.json
```
The scenario must explicitly set synthetic_only=true, real_trading=false,
commission_bps, slippage_bps, minimum_fee_pln and frames (1–370).
Each frame contains at, targets_generated_at (strictly earlier than at),
synthetic_quotes=true, quotes keyed by fixture: identifiers with bid, ask
and observed_at, and targets for all five A:global through E:global
accounts mapping fixture IDs to target PLN budgets. All prices are artificial.
The laboratory uses a new temporary SQLite ledger for each run, whole-share
sizing, reductions before purchases, explicit costs and the existing ledger
risk gates. Benchmark E holds its initial purchase. It rejects stale quotes,
missing quotes for held positions and unordered frames. No externally supplied
database path, no modification of live portfolio or persistent test balances.
Reports contain hypothetical executions, blocked attempts, equity history
and final account state. This is a scenario laboratory, not a historical
backtest, not a live equity paper runner, and not directly connected to
research_cycle targets. Sector and liquidity assumptions are not independently
enforced in this laboratory; scenario prices do not include actual exchange
sessions, market depth, FX execution, corporate actions or tax accounting.
Next gate: validate a real market data source and exchange calendars, then
design separate paper execution with actual fresh bid/ask, instrument eligibility,
persistent decisions and explicit compatibility/rollback plan.

## First actual provider preflight — 2026-11-27
Operator supplied successful terminal output at 09:29:57 UTC:
NBP USD/EUR 93 rates each; XNAS 84 sessions and 252 raw/252 adjusted bars
for AAPL, MSFT, QQQ; XNYS 84 sessions and 84 raw/84 adjusted bars for IBM;
last session 2026-11-25. ready=true, eligible_for_orders=false.
This confirms access to these endpoints and instruments, not independent
calendar verification, current executable quotes or broader entitlements.
New read-only scan:
```sh
python3 -m global_agent.equity_scan --prompt
```
It requests the same four instruments and provider calendars, converts
using prior NBP FX, validates complete sessions and emits retrospective
rankings by comparable venue cohort. No allocation, no persistent keys,
no raw market data publication, no ledger writes. Execute locally using
Paper credentials only; never share credentials or screenshots of them.
No scheduled data collection or automatic trade permissions yet.
