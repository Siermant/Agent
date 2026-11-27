# Multi-market data roadmap — research only
## v0.12: staged crypto expansion
Default GTA_CRYPTO_UNIVERSE=core, BTC-EUR and ETH-EUR. The Docker Compose files
remain identical to v0.11; activating the new image does not activate altcoin
orders. No database schema migration. A-E strategies and existing capital
unchanged. All mode is a separate configuration change and must be preceded
by an isolated eight-pair preflight. No downgrade to v0.11 after introducing
positions outside BTC/ETH; the old engine cannot value such positions.
Do not force liquidation or restore an old database to make a downgrade work.
Once all mode is activated, the current image-only updater must not be used
until compatibility with the active universe is checked in the updater.
All mode requires all eight fresh quotes and complete closed 20-day histories;
one missing pair blocks the whole crypto cycle. Existing same-day decisions
are preserved; newly enabled pairs can receive their first daily decision.
Benchmark E remains BTC. A-D budget per new position 4.5% of equity; ledger
limits unchanged. Eight pairs mean approximately 36% initial exposure at most
before price changes, not a guaranteed cap on market exposure.
## v0.13: equity/ETF data
Independent source adapter; no broker-order permissions and no credentials
in repository. First support daily completed sessions for research; do not
execute from delayed or stale data. Data model: venue/MIC, ISIN or stable
provider identifier, ticker, currency, exchange timezone, session date and
close timestamp, timestamp of retrieval, adjustment policy, provenance and
delay. Corporate actions (splits, dividends) must be handled before comparative
total-return research; actual executable prices distinct from adjusted series.
Market holidays, early closes and daylight saving must be validated by venue;
missing sessions are not filled by copying last price. NBP EUR and USD prior
reference rates for PLN research, not executable FX. Independent health per
market so crypto outage does not block equities and vice versa.
Candidate providers, official docs consulted 2026-11-27:
- IBKR: https://www.interactivebrokers.com/support/faq/content/obtain-historical-data-via-api-28236420
  https://www.interactivebrokers.com/support/faq/content/obtain-real-time-data-via-95554814
  Historical and live API data usually require appropriate market data subscription.
  Preferred candidate for eventual broker integration; no subscription purchased.
- Alpaca: https://docs.alpaca.markets/us/docs/market-data-faq
  https://docs.alpaca.markets/us/v1.4.2/docs/about-market-data-api
  US equity/ETF testing; basic live feed is IEX only, not consolidated SIP.
  Account eligibility, API access, redistribution terms and coverage to verify.
  US ETFs in research are not automatically eligible for EU retail/IKE.
- European UCITS ETFs and Polish IKE: provider, broker-supported instruments,
  license and venue coverage remain unverified. Use ISIN + listing identifier
  and instrument eligibility checks before simulated orders.
No paid subscriptions, live trading, leverage, options or new credentials
authorized by this document. Offline CSV fixtures are the next development
step, then provider integration and read-only preflight. IKE tax treatment
requires independent verification. This is not an investment recommendation.
