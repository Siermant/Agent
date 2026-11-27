"""One-time, local paper-only opening of Global A-D. No broker orders.
All five portfolios are independent hypothetical alternatives. E stays cash.
The database is published only after all initial fills and metadata commit.
"""
import argparse
import fcntl
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
from contextlib import closing

from .ledger import Ledger, CAPITALS, EXPOSURE
from .equity_data import moment
from .equity_job import credentials, QUALIFIED
from .equity_quotes import clients, qualify, SYMBOLS
from .global_preview import preview, SECTORS
from .equity_expand import asset_client
from .equity_preflight import calendar_client
from .alpaca_data import client

ACCOUNTS=tuple(p+':global' for p in 'ABCDE')
CAPITAL=CAPITALS['global']
RATE=D('.001')
SLIP=D('.001')
MIN_FEE=D('2')
CENT=D('.01')
IDS={f'alpaca:{mic}:{s}':s for mic,ss in QUALIFIED.items() for s in ss if s in SYMBOLS}


def number(v):
    n=D(str(v))
    if not n.is_finite() or n<0:
        raise ValueError('Invalid plan amount')
    return n


def validate(plan,now):
    if (plan.get('research_only') is not True or plan.get('real_trading') is not False
        or plan.get('eligible_for_orders') is not False or plan.get('initial_cash_only') is not True
        or plan.get('aligned_us_research') is not True or plan.get('state')!='allocation_preview'):
        raise ValueError('Expected initial research plan')
    if not timedelta(0)<=now-moment(plan['generated_utc'])<=timedelta(hours=24):
        raise ValueError('Plan not fresh')
    cost=plan['cost_scenario']
    if [number(cost[k]) for k in ('commission_bps','spread_slippage_bps','minimum_fee_pln','minimum_adv_pln','max_adv_fraction')]!=[D(10),D(10),MIN_FEE,D(100000),D('.001')]:
        raise ValueError('Unexpected cost scenario')
    eligible={}
    seen=set()
    for r in plan['instruments']:
        ident=f"alpaca:{r['mic']}:{r['ticker']}"
        if ident in seen:
            raise ValueError('Duplicate instrument metadata')
        seen.add(ident)
        if ident in IDS and r.get('research_eligible') is True and r['sector_assumed']==SECTORS[IDS[ident]]:
            adv=number(r['iex_turnover_20_pln'])
            if adv>=D(100000):
                eligible[ident]=adv
    rows=plan['plans']
    if len(rows)!=5 or {r['account'] for r in rows}!=set(ACCOUNTS):
        raise ValueError('Expected five portfolios')
    out={}
    for row in rows:
        account=row['account']
        p=account[0]
        if number(row['capital_pln'])!=CAPITAL:
            raise ValueError('Unexpected starting capital')
        budgets={}
        sectors={}
        principal=fees=D(0)
        for a in row['allocations']:
            ident=a['instrument_id']
            if p=='E' or ident not in eligible or ident in budgets or a['sector']!=SECTORS[IDS[ident]]:
                raise ValueError('Unexpected allocation')
            notional=number(a['notional_pln'])
            cost=number(a['estimated_entry_cost_pln'])
            budget=notional+cost
            if notional<=0 or budget>min(CAPITAL*D('.045'),eligible[ident]*D('.001')):
                raise ValueError('Position budget exceeded')
            budgets[ident]=budget
            principal+=notional
            fees+=cost
            sectors[a['sector']]=sectors.get(a['sector'],D(0))+budget
        if (number(row['allocated_pln'])!=principal or number(row['estimated_cost_pln'])!=fees
                or number(row['cash_pln'])+principal+fees!=CAPITAL
                or number(row['cash_pln'])<CAPITAL*D('.20')
                or (p!='E' and principal+fees>CAPITAL*EXPOSURE[p])
                or any(v>CAPITAL*D('.20') for v in sectors.values())):
            raise ValueError('Portfolio budget exceeded')
        out[account]=budgets
    return out


def account_summary(ledger):
    return [dict(s,positions={k:str(v) for k,v in s['positions'].items()})
            for s in ledger.status() if s['account'] in ACCOUNTS]


def read_existing(path):
    # Read-only SQLite: do not recreate a lost/corrupt database or reset capital.
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        row=db.execute('SELECT payload FROM initial_receipt WHERE id=1').fetchone()
        if row is None:
            raise ValueError('Missing initial receipt')
        result=json.loads(row[0])
        if result.get('real_trading') is not False or result.get('state')!='opened':
            raise ValueError('Invalid initial receipt')
    return dict(result,state='already_opened')


def open_initial(root,plan,snapshot,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc))
    root=Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Dedicated state directory required')
    path=root/'global-paper.db'
    with (root/'initial.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            return read_existing(path)
        now=clock()
        budgets=validate(plan,now)
        if (snapshot.get('ready') is not True or snapshot.get('market_open') is not True
            or snapshot.get('research_only') is not True or snapshot.get('real_trading') is not False
            or snapshot.get('eligible_for_orders') is not False or snapshot.get('state')!='qualified'
            or snapshot.get('source')!='alpaca_iex_latest' or snapshot.get('fx_source')!='nbp_table_a_prior_day'
            or not timedelta(0)<=now-moment(snapshot['checked_utc'])<=timedelta(seconds=15)):
            raise ValueError('Fresh qualified snapshot required')
        close=moment(snapshot['session_close_utc'])
        if close-now<timedelta(seconds=90):
            raise ValueError('Too close to end of session')
        if set(snapshot['quotes'])!=set(SYMBOLS):
            raise ValueError('Missing quote')
        quotes={}
        sizes={}
        for ident,s in IDS.items():
            q=snapshot['quotes'][s]
            bid,ask=number(q['bid_pln_reference']),number(q['ask_pln_reference'])
            stamp=moment(q['observed_at'])
            size=q['ask_size_shares']
            if (not 0<bid<=ask or (ask-bid)/bid>D('.01') or
                not timedelta(0)<=now-stamp<=timedelta(seconds=60) or type(size) is not int or size<=0):
                raise ValueError('Invalid quote')
            quotes[ident]=(bid,ask,stamp)
            sizes[ident]=D(size)
        fills=[]
        skipped=[]
        with tempfile.TemporaryDirectory(dir=root,prefix='initial-') as tmp:
            temp=Path(tmp)/'ledger.db'
            ledger=Ledger(temp,clock=clock)
            for account in ACCOUNTS:
                ledger.mark(account,quotes)
                for ident,budget in budgets[account].items():
                    price=quotes[ident][1]*(1+SLIP)
                    qty=min(sizes[ident],max(D(0),((budget-MIN_FEE-CENT)/(price*(1+RATE))).to_integral_value(rounding=ROUND_DOWN)))
                    if qty<1:
                        skipped.append({'account':account,'instrument_id':ident,'reason':'whole_share_or_displayed_size'})
                        continue
                    fee=max(MIN_FEE,(qty*price*RATE).quantize(CENT,rounding=ROUND_HALF_UP))
                    if qty*price+fee>budget or clock()>=close:
                        raise ValueError('Budget or session expired')
                    r=ledger.execute(account,'initial:'+ident,ident,'buy',qty,quotes,RATE,SLIP,MIN_FEE,market_open=True,approved=True)
                    fills.append(dict(r,account=account,instrument_id=ident,quantity=str(qty)))
                ledger.mark(account,quotes)
            if not fills:
                raise ValueError('No simulated initial fills')
            receipt={'state':'opened','research_only':True,'real_trading':False,'eligible_for_orders':False,
                     'simulated':True,'one_time_initial_only':True,'rebalance_enabled':False,
                     'generated_utc':clock().isoformat(),'plan_generated_utc':plan['generated_utc'],
                     'fx_date':snapshot['fx_date'],'fx_reference_only':True,
                     'quote_source':'alpaca_iex_latest','cost_scenario':'10bps commission min 2 PLN + 10bps on ask',
                     'fills':fills,'skipped':skipped,'accounts':account_summary(ledger)}
            with ledger.tx() as db:
                db.execute('CREATE TABLE initial_receipt(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL)')
                db.execute('INSERT INTO initial_receipt VALUES(1,?)',(json.dumps(receipt,allow_nan=False),))
            with closing(sqlite3.connect(temp,isolation_level=None)) as db:
                if db.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]!=0:
                    raise ValueError('Could not checkpoint ledger')
                if db.execute('PRAGMA journal_mode=DELETE').fetchone()[0]!='delete':
                    raise ValueError('Could not consolidate ledger')
            with temp.open('rb') as f:
                os.fsync(f.fileno())
            os.chmod(temp,0o600)
            os.link(temp,path)
            fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--apply-paper',action='store_true',required=True)
    p.parse_args()
    try:
        root=Path(os.environ['STATE_DIRECTORY'])
        if root.name!='gta-global-paper':
            raise ValueError('Unexpected isolated state directory')
        path=root/'global-paper.db'
        if path.exists():
            r=read_existing(path)
        else:
            key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
            ck,qt=clients(key,secret)
            first=qualify(ck,qt)
            if not first['ready']:
                print(json.dumps(first,indent=2))
                raise SystemExit(0 if first['state']=='market_closed' else 2)
            plan=preview(asset_client(key,secret),calendar_client(key,secret),client(key,secret))
            snapshot=qualify(ck,qt,include_quotes=True)
            r=open_initial(root,plan,snapshot)
        print(json.dumps(r,indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'state':'failed','error_type':type(exc).__name__}))
        raise SystemExit(1)

if __name__=='__main__':
    main()
