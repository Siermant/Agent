"""Price-only Global A-E valuation of the existing isolated paper ledger.
Never creates accounts, changes positions, executes orders or touches crypto.
Corporate actions and dividend cash are not yet accounted for.
"""
import json
import os
import fcntl
import sqlite3
from contextlib import closing
from datetime import datetime,timedelta,timezone
from decimal import Decimal as D
from pathlib import Path
from .global_paper_initial import ACCOUNTS,CAPITAL,IDS,read_existing,number
from .equity_data import moment
from .equity_job import credentials
from .equity_quotes import clients,qualify
from .ledger import Ledger
from .research_cycle import publish
from .global_panel import export_panel

FLAGS={'research_only':True,'real_trading':False,'eligible_for_orders':False,
       'simulated':True,'rebalance_enabled':False,'corporate_actions_verified':False,
       'return_type':'price_only_reference_fx'}

def read_holdings(path):
    receipt=read_existing(path)
    expected={r['account']:r for r in receipt['accounts']}
    if set(expected)!=set(ACCOUNTS) or len(receipt['accounts'])!=5:
        raise ValueError('Invalid opening receipt')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        actual={}
        for account in ACCOUNTS:
            r=db.execute('SELECT initial,cash FROM accounts WHERE id=?',(account,)).fetchone()
            pos=dict(db.execute('SELECT symbol,qty FROM positions WHERE account=?',(account,)))
            fills=db.execute('SELECT count(*) FROM fills WHERE account=?',(account,)).fetchone()[0]
            if (r is None or number(r[0])!=CAPITAL or number(r[1])!=number(expected[account]['cash_pln'])
                or {k:number(v) for k,v in pos.items()}!={k:number(v) for k,v in expected[account]['positions'].items()}
                or fills!=expected[account]['fills'] or not set(pos)<=set(IDS)
                or any(number(q)<=0 for q in pos.values())):
                raise ValueError('Positions differ from initial experiment')
            actual[account]={'cash':number(r[1]),'positions':{k:number(v) for k,v in pos.items()},'fills':fills}
        return actual

def mark(root,fetch,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc))
    root=Path(root).resolve()
    path=root/'global-paper.db'
    # Same lock as initial opening; never create an empty ledger.
    with (root/'initial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not path.is_file():
            return dict(FLAGS,state='waiting_for_initial_open',checked_utc=clock().isoformat())
        holdings=read_holdings(path)
        selected=sorted({IDS[k] for r in holdings.values() for k in r['positions']})
        if not selected:
            raise ValueError('No held instruments')
        snapshot=fetch(selected)
        now=clock()
        base=dict(FLAGS,checked_utc=now.isoformat())
        if snapshot.get('ready') is not True:
            return dict(base,state=snapshot.get('state','quotes_unavailable'))
        if (snapshot.get('state')!='qualified' or snapshot.get('market_open') is not True
            or snapshot.get('research_only') is not True or snapshot.get('real_trading') is not False
            or snapshot.get('eligible_for_orders') is not False
            or snapshot.get('source')!='alpaca_iex_latest' or snapshot.get('fx_source')!='nbp_table_a_prior_day'
            or not timedelta(0)<=now-moment(snapshot['checked_utc'])<=timedelta(seconds=15)
            or now>=moment(snapshot['session_close_utc']) or set(snapshot['quotes'])!=set(selected)):
            raise ValueError('Invalid valuation snapshot')
        bids={}
        for s,q in snapshot['quotes'].items():
            bid,ask=number(q['bid_pln_reference']),number(q['ask_pln_reference'])
            if not 0<bid<=ask or (ask-bid)/bid>D('.01') or not timedelta(0)<=now-moment(q['observed_at'])<=timedelta(seconds=60):
                raise ValueError('Stale or invalid valuation quote')
            bids[s]=bid
        rows=[]
        # Existing receipt and holdings validated under the common lock.
        # Use Ledger snapshot risk logic in one SQLite transaction for all five.
        # Construct without __init__: it would initialize accounts on missing data.
        ledger=Ledger.__new__(Ledger)
        ledger.path=path
        with ledger.tx() as db:
            for account in ACCOUNTS:
                row,pos=ledger._account(db,account)
                h=holdings[account]
                if pos!=h['positions'] or number(row['cash'])!=h['cash']:
                    raise ValueError('Concurrent holdings change')
                invested=sum((qty*bids[IDS[k]] for k,qty in pos.items()),D(0))
                equity=h['cash']+invested
                blocked,day,week,peak=ledger._snapshot(db,row,equity,now)
                rows.append({'account':account,'capital_pln':str(CAPITAL),
                             'cash_pln':str(h['cash']),'invested_pln':str(invested),
                             'equity_pln':str(equity),'pnl_pln':str(equity-CAPITAL),
                             'return_pct':str((equity/CAPITAL-1)*100),
                             'drawdown_pct':str((1-equity/peak)*100),'risk_blocked':blocked,
                             'fills':h['fills'],'positions_count':len(pos),
                             'positions':[{'ticker':IDS[k],'quantity':str(qty),'value_pln':str(qty*bids[IDS[k]])} for k,qty in sorted(pos.items())]})
            fills=[{'account':r['account'],'ticker':IDS[r['symbol']],'quantity':r['qty'],
                    'price_pln':r['price'],'fee_pln':r['fee'],'at':r['at'],'side':r['side']}
                   for r in db.execute("SELECT * FROM fills WHERE account LIKE '%:global' ORDER BY at DESC,account,order_id LIMIT 200")]
            # Keep full history in SQLite, export last 30 UTC days' final common marks.
            history=[]
            cutoff=(now-timedelta(days=30)).date().isoformat()
            marks=db.execute('SELECT substr(at,1,10) AS d, max(at) AS stamp FROM snapshots WHERE account=? AND substr(at,1,10)>=? GROUP BY d ORDER BY d',('A:global',cutoff)).fetchall()
            for d,stamp in marks:
                vals={r['account']:r['equity'] for r in db.execute('SELECT account,equity FROM snapshots WHERE at=?',(stamp,)) if r['account'] in ACCOUNTS}
                if set(vals)==set(ACCOUNTS):
                    history.append({'at':stamp,'equity_pln':vals})
        return dict(base,state='valued',valued_at=now.isoformat(),fx_date=snapshot['fx_date'],
                    quote_source='alpaca_iex_latest',accounts=rows,history=history,fills=fills)

def main():
    root=Path(os.environ['STATE_DIRECTORY'])
    if root.name!='gta-global-paper':
        raise SystemExit('Unexpected isolated state directory')
    def fetch(symbols):
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        return qualify(*clients(key,secret,symbols=symbols),symbols=symbols,include_quotes=True)
    try:
        result=mark(root,fetch)
        if result['state']=='valued':
            publish(result,root/'valuation.json')
        status={k:v for k,v in result.items() if k not in ('accounts','history','fills')}
        publish(status,root/'valuation-status.json')
        cache=os.getenv('CACHE_DIRECTORY')
        if cache:
            previous=None
            if result['state']=='valued':
                previous=result
            elif (root/'valuation.json').is_file():
                previous=json.loads((root/'valuation.json').read_text())
            export_panel(cache,status,previous)
        print(json.dumps({k:v for k,v in result.items() if k not in ('history','fills')},allow_nan=False))
    except Exception as exc:
        error=dict(FLAGS,state='failed',checked_utc=datetime.now(timezone.utc).isoformat(),error_type=type(exc).__name__)
        publish(error,root/'valuation-status.json')
        cache=os.getenv('CACHE_DIRECTORY')
        if cache:
            export_panel(cache,error,None)
        print(json.dumps(error))
        raise SystemExit(1)

if __name__=='__main__':
    main()
