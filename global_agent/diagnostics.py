"""Paper-only statistics and allowlisted diagnostic export (no raw logs/secrets)."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from io import BytesIO
import json
import sqlite3
from zipfile import ZipFile, ZIP_DEFLATED


def read_db(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def statistics(ledger, now):
    """Consistent read snapshot; return since inception, drawdown from peak, 30 UTC days."""
    cutoff = (now.date() - timedelta(days=29)).isoformat()
    out = []
    with closing(read_db(ledger.path)) as db:
        db.execute('BEGIN')
        for a in db.execute('SELECT * FROM accounts ORDER BY id'):
            account = a['id']
            initial = D(a['initial'])
            cash = D(a['cash'])
            last = db.execute('SELECT at,equity FROM snapshots WHERE account=? ORDER BY at DESC LIMIT 1', (account,)).fetchone()
            fills = list(db.execute('SELECT side,fee,qty,price FROM fills WHERE account=?', (account,)))
            # Last valuation on each recorded day, not reconstructed missing values.
            history = [dict(r) for r in db.execute('''SELECT substr(s.at,1,10) AS day,s.at,s.equity
                FROM snapshots s JOIN (SELECT max(at) AS at FROM snapshots WHERE account=? AND at>=? GROUP BY substr(at,1,10)) d
                ON s.at=d.at WHERE s.account=? ORDER BY s.at''', (account, cutoff, account))]
            value = D(last['equity']) if last else None
            at = datetime.fromisoformat(last['at']) if last else None
            fresh = bool(at and timedelta(0) <= now-at <= timedelta(seconds=90))
            peak = D(a['peak'])
            out.append({'account':account,'initial_pln':str(initial),'cash_pln':str(cash),
                'equity_pln':str(value) if value is not None else None, 'valued_at':last['at'] if last else None,
                'fresh':fresh,'return_pct':str((value/initial-1)*100) if value is not None else None,
                'current_drawdown_pct':str((1-value/peak)*100) if value is not None else None,
                'fees_pln':str(sum((D(r['fee']) for r in fills),D(0))), 'fills':len(fills),
                'turnover_pln':str(sum((D(r['qty'])*D(r['price']) for r in fills),D(0))), 'history':history})
    return out


def decision_counts(ledger):
    with closing(read_db(ledger.path)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_decisions'").fetchone():
            return []
        # Only known outcome codes; do not export arbitrary strings or raw logs.
        allowed = {'hold','no_signal','benchmark_hold','risk_blocked','no_budget','filled_buy','filled_sell','recovered_fill'}
        return [{'account':r['account'],'result':r['result'] if r['result'] in allowed else 'other','count':r['n']}
                for r in db.execute('SELECT account,result,count(*) AS n FROM paper_decisions GROUP BY account,result ORDER BY account,result')]


def report(ledger, store, state, now=None):
    now = now or datetime.now(timezone.utc)
    market = store.status(now)
    # Explicit allowlist; no environment, paths, logs, credentials, hostnames, IPs or raw DBs.
    info = {'generated_utc':now.isoformat(),'version':state['version'],'mode':state['mode'],
            'real_trading':False,'market_healthy':market['healthy'],'market_checked':market['checked'],
            'fresh_symbols':[q['symbol'] for q in market['quotes'] if q['fresh']],
            'paper_enabled':bool(state['paper']['enabled']), 'paper_state':state['paper']['state'],
            'note':'Simulated portfolios only. Last valuations may be stale; no taxes. Not investment performance evidence.'}
    buf = BytesIO()
    with ZipFile(buf,'w',ZIP_DEFLATED) as z:
        for name,data in [('status.json',info),('portfolios.json',statistics(ledger,now)),('decisions.json',decision_counts(ledger))]:
            z.writestr(name,json.dumps(data,ensure_ascii=False,indent=2))
    return buf.getvalue()
