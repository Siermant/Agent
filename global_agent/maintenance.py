"""Bound high-frequency valuation history without changing ledger balances.
Retain every point for the last 7 UTC dates, daily close thereafter.
Preserve each account's lifetime peak and period anchors in accounts.
Never delete executions or decision history.
"""
from datetime import timedelta
from .ledger import utc


def compact_snapshots(ledger, now, retain_days=7):
    if not isinstance(retain_days, int) or retain_days < 2:
        raise ValueError('At least two UTC dates must remain uncompressed')
    cutoff = (utc(now).date() - timedelta(days=retain_days-1)).isoformat()
    with ledger.tx() as db:
        # Only past days; no reconstruction or averaging of price data.
        cursor = db.execute("""DELETE FROM snapshots WHERE at < ? AND rowid NOT IN (
            SELECT s.rowid FROM snapshots s JOIN (
                SELECT account, max(at) AS at FROM snapshots WHERE at < ?
                GROUP BY account, substr(at,1,10)
            ) keep USING(account,at))""", (cutoff, cutoff))
        return cursor.rowcount
