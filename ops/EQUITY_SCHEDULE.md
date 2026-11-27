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

## Qualified daily universe (operator output, 2026-11-27 09:39 UTC)
28 of 30 candidates ranked successfully in 22.133 s; peak RAM 21.2 MiB
(as reported by systemd). 14 XNAS, 14 XNYS. SPY and VT excluded due to
unsupported venue calendars. No historical-series exclusions. The expanded
daily job re-discovers asset metadata and requires exactly these listing
memberships and exclusions. Changed venue, inactive asset, missing row,
duplicate row or changed exclusion fails closed; retains previous ranking.
No change to credentials, timer, service unit or ledger. Run full tests on
staged root-owned checkout before changing code. Stop timer and service,
retain previous directory, switch checkout, run service once and verify
success and 28 instruments before restarting timer. Restore previous code
if validation fails; retained status/ranking may reflect the failed attempt
until previous service successfully reruns. No live broker orders.

## Dashboard integration (staged, not installed)
Operator confirmed successful 28-instrument daily service update, 127/127
VM tests, 09:44:27 UTC on 2026-11-27. The local HTTP dashboard and status
API now support optional GTA_EQUITY_REPORT_DIR. Reader validates flags,
counts, timestamp agreement, bounded size and numeric metrics; reports
older than 30 hours are stale. This limit concerns report generation, not
the age of the latest market session. HTML is escaped. No rank data
included in remote diagnostics by this change.
No Compose changes yet. The production systemd state directory is private
and belongs to the independent research job. Do not weaken its permissions
or mount /etc/gta-secrets. Next: publish a separate limited read-only
dashboard export with an atomic snapshot and explicit stale/failure state,
then review the Compose mount, qualify a new image and activate with rollback.
The optional reader remains disconnected until that bridge is installed.

## v0.14 panel bridge — staging
Research unit adds CacheDirectory=gta-equity-export (0755), with an
atomic single panel.json export (0644). The private StateDirectory and
credential permissions are unchanged. The export contains only allowed
status fields, instrument identifiers, ranking metrics and session labels;
no raw bars, credentials or ledger. On scan start/failure publish a
status-only snapshot. An export I/O failure may leave an older snapshot;
the reader expires reports after 30 hours. Systemd DynamicUser normally
places the cache under /var/cache/private with a symlink; the Docker daemon
must resolve /var/cache/gta-equity-export and UID 10001 must be able to read
the mounted panel.json. Verify this on the actual VM.
Optional compose.equity.yaml binds only this export directory read-only,
never private state or /etc/gta-secrets. Existing image-only update
tool is deliberately not used to change Compose. Stage host export first,
then verify candidate offline and mount in a separate container. A
production Compose migration and rollback procedure are still required.
No current trading, data sharing with remote diagnostics, or global
ledger changes. Provider data remains limited to the qualified 28 US
instruments; calendar independence and trade eligibility remain unverified.

## v0.14 guarded migration
The migration tool ops/migrate_equity_panel.py accepts a fresh (<=24h)
successful qualification manifest for candidate 74b1c5638414e46eaa599cf01a1568773ffc1204 only.
It checks the existing v0.13 core BTC/ETH service and enabled timer, stages
and tests reviewed code, runs crypto preflight, then switches the research
service and verifies its actual production export with the candidate image.
Only then stops the crypto container, archives the named volume with CRC
verification, tags the previous image, changes COMPOSE_FILE to add the
reviewed read-only overlay, and starts the candidate. Verifies 15 accounts,
fill count, health, remote diagnostic readiness, 28 equity rankings and
actual read-only mount. On caught exception, attempts to restore previous
image, research code, unit, environment and timer. No database restore or
volume deletion. Retains recovery note, previous code and archive.
Rollback is best effort: power loss/SIGKILL, rollback command failures,
or filesystem failure require manual recovery. No successful VM migration
has been confirmed yet. Do not use the older image-only updater after
this Compose migration; its explicit two-file configuration would omit the
equity overlay. Adapt and qualify future updater before next image change.
The qualification candidate does not include the migration tool; fetch the
separately reviewed installer commit to a temporary root-owned directory.

## Operator confirmation and next Global gate
Operator reported v0.14 installed. The isolated global_preview returned
allocation_preview on 2026-11-27 at 10:20 UTC with 28 instruments, 26 stocks
research-eligible, two ETFs deliberately excluded and benchmark E in cash.
The actual portfolio plans are hypothetical, not fills. The IEX turnover
proxy is not consolidated liquidity and all costs remain scenario inputs.
The separate equity_quotes module checks the Paper US market clock twice
and latest IEX bid/ask with prior-day NBP USD FX. It accepts quotes up to
60 seconds old, positive bid/ask/sizes, uncrossed spread <=1%, same NY date.
All 26 stocks must pass for the qualification to be ready; closed sessions
return market_closed with ready=false, not a successful quote qualification.
No raw quotes printed, no persistent ledger or orders. The clock and quote
endpoints still need actual VM validation during regular market hours.
Reference: https://docs.alpaca.markets/us/reference/legacyclock ;
https://docs.alpaca.markets/us/reference/stocklatestquotes-1 .
