# Independent equity research job (VM 106)
A separate host systemd oneshot service, no Docker changes, no ledger writes,
no live orders. Daily 07:15 Europe/Warsaw, including weekends; prior
completed US sessions only. Persistent timer catches missed starts after
reboot. No intraday signals, no automatic portfolio allocation.
Paper credentials stored unencrypted at rest in root-only
/etc/gta-secrets/alpaca-paper.json (0600), outside Git. Systemd LoadCredential
passes a private runtime copy to the DynamicUser service. The provisioner
never echoes or overwrites the credentials; do not include them in backups
or reports. Rotate using a separate reviewed procedure if necessary.
Root-owned isolated code at /opt/gta-equity-research; never run a moving
branch or user-writable checkout as a service. Use the approved full SHA.
Systemd unit limits memory to 192 MiB and CPU to 50% of one core, timeout
10 minutes. No network sandbox beyond allowed address families; application
code uses fixed GET endpoints, no brokerage-order methods. Paper API keys
may have paper order permissions at provider level.
Status and last successful ranking stay local under
/var/lib/gta-equity-research (root access). On failed/incomplete scan,
ranking.json remains the previous successful result; status.json records
failure with timestamp and redacted error type. A killed or timed-out
process may leave state=running; check systemd and timestamps. Never
interpret file presence as current successful analysis. No data uploaded.
Before enabling timer, verify units and execute a one-time service start
successfully. Inspect status, then enable timer. To disable recurring
execution: systemctl disable --now gta-equity-research.timer
To stop an in-progress scan separately: systemctl stop gta-equity-research.service
No changes to existing crypto data or service. Installation and actual
scheduled execution not yet tested on the VM. Initial universe remains
AAPL/MSFT/QQQ and IBM; prior manual ranking output not yet supplied.

## Operator confirmation and expansion qualification
27.11.2026 09:35 UTC: operator confirmed 120 tests on VM, successful
first scheduled-service manual start, four ranked instruments, two cohorts,
and enabled timer next due 28.11.2026 07:15 CET. Ranking received, all four
instruments present. No change to active service.
equity_expand is an independent one-time discovery/scan of 30 US research
candidates. IBM is retained; GLD is deferred to commodity research. Asset
metadata is obtained by GET from the Paper endpoint; only active us_equity
assets on supported NASDAQ/NYSE calendar venues are included. Other venues
(e.g. NYSE Arca) are explicitly excluded, not mapped to another exchange.
Current listing metadata is not point-in-time historical identity verification.
ETF/stock labels are curated, not inferred from the asset class.
The test uses existing systemd credentials without copying secrets.
Run from separate root-owned qualification code, not the scheduled checkout,
using a transient DynamicUser unit with LoadCredential. No changes to timer,
no persistent ranking, no ledger access through application code, no orders.
The expanded universe is not yet qualified against actual market data.
