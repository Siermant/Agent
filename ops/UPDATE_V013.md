# v0.13 research-only candidate
Image-only update; Compose and existing database schema unchanged. Existing BTC/ETH paper runner stays active, Global and IKE accounts remain inactive. No automatic equity orders.
New modules: equity_data, fx_reference, alpaca_data, equity_rank, global_plan, research_cycle, global_sim, equity_preflight.
After updating the image with the approved full commit SHA and existing updater:
```sh
docker compose exec agent python -m global_agent.equity_preflight --prompt
```
Create an Alpaca Paper Only Account if desired; do not fund it or use live credentials. Enter the paper key and secret in the local interactive terminal (hidden). Do not send credentials in chat, GitHub, command-line arguments or screenshots. Keys are held in memory for this check only. The program makes GET requests to fixed paper-calendar and market-data endpoints, no redirects, no orders. It tests the last 120 calendar days of XNAS (AAPL/MSFT/QQQ) and XNYS (IBM), raw and split/dividend-adjusted IEX bars and NBP reference FX. Output contains counts, latest session and redacted error types/status codes, not prices or keys. Provider calendar is not independently verified against the exchanges. No persistent data-source activation, no schedule, no trade eligibility, no changes to existing ledger.
Official docs:
https://docs.alpaca.markets/us/docs/paper-trading
https://docs.alpaca.markets/us/reference/calendar-2
https://docs.alpaca.markets/us/reference/stockbars
https://api.nbp.pl/en.html
Actual API entitlements and response formats have not been verified using real credentials; test fixtures are synthetic. The image must pass full offline qualification and existing crypto network preflight before activation. Do not enable extended crypto universe during this update.
