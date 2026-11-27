"""Read-only Global rebalancing proposals for the initial-only experiment.
Targets come from the existing ranking model. A rank is not a return forecast.
No ledger writes, orders, position changes, or claim of executable prices.
"""
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime,timedelta,date,timezone
from decimal import Decimal as D,ROUND_DOWN,ROUND_HALF_UP
from pathlib import Path
from .alpaca_data import NY,client as bars_client
from .equity_data import moment
from .equity_job import credentials
from .equity_quotes import clients,qualify,SYMBOLS
from .equity_expand import asset_client
from .equity_preflight import calendar_client
from .global_preview import preview as initial_preview,SECTORS
from .global_paper_initial import ACCOUNTS,IDS,RATE,SLIP,MIN_FEE,CENT,read_existing,validate,number
from .global_paper_mark import read_holdings
from .ledger import EXPOSURE
from .corporate_actions import scan,client as corporate_client
from .corporate_audit import validate as validate_feed,digest

FLAGS={'research_only':True,'real_trading':False,'eligible_for_orders':False,
       'rebalance_enabled':False,'simulated':True,'accounting_enabled':False,
       'expected_return_verified':False,'corporate_actions_verified':False}
# Conservative experimental assumptions, not changes to execution risk limits.
COOLDOWN=timedelta(days=7)
MAX_TURNOVER=D('.15')  # sum of gross purchases and sales / current equity
MIN_WEIGHT=D('.01')
EVENT_LOOKAHEAD=timedelta(days=7)

def read_audit(path,now):
    path=Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.execute('BEGIN')
        r=db.execute('SELECT id,at,start,end,digest FROM scans ORDER BY at DESC LIMIT 1').fetchone()
        if r is None or not timedelta(0)<=now-moment(r[1])<=timedelta(hours=36):
            raise ValueError('Fresh corporate audit scan required')
        sid,stamp,start,end,stored=r
        events={}
        hashes={}
        for ident,h,body in db.execute('SELECT o.event_id,o.digest,r.payload FROM observations o JOIN revisions r USING(event_id,digest) WHERE o.scan_id=?',(sid,)):
            e=json.loads(body)
            if digest(e)!=h or e['id']!=ident:
                raise ValueError('Audit revision hash mismatch')
            events[ident]=e;hashes[ident]=h
        if digest({'start':start,'end':end,'events':hashes})!=stored:
            raise ValueError('Audit scan hash mismatch')
        # Preserve flags across future scans: changed revisions are not approved
        # by merely seeing the same revised value again.
        unresolved={}
        for ident,h,body in db.execute('SELECT event_id,digest,payload FROM revisions'):
            e=json.loads(body)
            if digest(e)!=h or e['id']!=ident:
                raise ValueError('Audit history hash mismatch')
            if ident not in hashes or hashes[ident]!=h:
                unresolved.setdefault(ident,[]).append(e)
        return {'checked_utc':stamp,'hashes':hashes,'events':events,'unresolved':unresolved}

def event_blocks(report,audit,opened,today):
    _,_,events=validate_feed(report)
    blocks={}
    def add(s,reason):
        blocks.setdefault(s,set()).add(reason)
    def relevant(e):
        dates=e['dates']
        key=dates.get('ex_date') or dates.get('effective_date')
        if key is None:return True
        d=date.fromisoformat(key)
        # Past entitlement before the initial-only experiment is irrelevant
        # for cash dividends; non-cash identity changes are conservative.
        if e['type']=='cash_dividends' and d<=opened:return False
        return d<=today+EVENT_LOOKAHEAD
    for ident,e in events.items():
        if not relevant(e):continue
        for s in e['symbols']:
            if e['type']!='cash_dividends':
                add(s,'noncash_event_review')
            elif e.get('state')!='simple_dividend_candidate':
                add(s,'unresolved_dividend')
            else:
                # Even explicit USD dividends are not posted in this experiment.
                add(s,'dividend_not_accounted')
            if audit['hashes'].get(ident)!=digest(e):
                add(s,'new_or_revised_event')
    for e in audit['events'].values():
        if e['id'] not in events and relevant(e):
            for s in e['symbols']:add(s,'event_missing_from_current_feed')
    for versions in audit['unresolved'].values():
        for e in versions:
            if relevant(e):
                for s in e['symbols']:add(s,'unresolved_event_history')
    return {s:sorted(reasons) for s,reasons in sorted(blocks.items())}

def read_state(path):
    path=Path(path).resolve()
    receipt=read_existing(path)
    holdings=read_holdings(path)
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        for account in ACCOUNTS:
            r=db.execute('SELECT peak,day,day_equity,week,week_equity FROM accounts WHERE id=?',(account,)).fetchone()
            last=db.execute('SELECT at,equity FROM snapshots WHERE account=? ORDER BY at DESC LIMIT 1',(account,)).fetchone()
            if r is None or last is None:raise ValueError('Missing risk history')
            holdings[account]['risk']=dict(r)
            holdings[account]['last']=dict(last)
    return receipt,holdings

def fee(value):
    return max(MIN_FEE,(value*RATE).quantize(CENT,rounding=ROUND_HALF_UP))

def propose(receipt,holdings,plan,snapshot,blocks,now):
    budgets=validate(plan,now)
    if (snapshot.get('ready') is not True or snapshot.get('market_open') is not True
        or snapshot.get('state')!='qualified' or snapshot.get('source')!='alpaca_iex_latest'
        or snapshot.get('fx_source')!='nbp_table_a_prior_day'
        or snapshot.get('real_trading') is not False or snapshot.get('eligible_for_orders') is not False
        or snapshot.get('research_only') is not True
        or not timedelta(0)<=now-moment(snapshot['checked_utc'])<=timedelta(seconds=15)
        or moment(snapshot['session_close_utc'])-now<timedelta(seconds=90)
        or set(snapshot['quotes'])!=set(SYMBOLS)):
        raise ValueError('Fresh full-universe quotes required')
    opened=moment(receipt['generated_utc'])
    if opened>now or set(holdings)!=set(ACCOUNTS):
        raise ValueError('Invalid holdings chronology')
    quotes={}
    for k,s in IDS.items():
        q=snapshot['quotes'][s]
        bid,ask=number(q['bid_pln_reference']),number(q['ask_pln_reference'])
        if not 0<bid<=ask or (ask-bid)/bid>D('.01') or not timedelta(0)<=now-moment(q['observed_at'])<=timedelta(seconds=60):
            raise ValueError('Invalid quotes')
        bs,az=q['bid_size_shares'],q['ask_size_shares']
        if any(type(n) is not int or n<=0 for n in (bs,az)):
            raise ValueError('Invalid displayed size')
        quotes[k]=(bid,ask,D(bs),D(az))
    adv={f"alpaca:{r['mic']}:{r['ticker']}":number(r['iex_turnover_20_pln']) for r in plan['instruments']}
    rows=[]
    for a in ACCOUNTS:
        p=a[0]
        h=holdings[a]
        pos=dict(h['positions'])
        cash=h['cash']
        if not set(pos)<=set(IDS) or any(q<=0 or q!=q.to_integral_value() for q in pos.values()):
            raise ValueError('Invalid held shares')
        def value(positions):
            return sum((q*quotes[k][0] for k,q in positions.items()),D(0))
        eq=cash+value(pos)
        row={'account':a,'equity_pln':str(eq),'proposed_changes':[],
             'state':'no_change','estimated_cost_pln':'0','turnover_pln':'0'}
        rows.append(row)
        if p=='E':
            row['state']='benchmark_waiting'
            continue
        if now-opened<COOLDOWN:
            row['state']='cooldown'
            continue
        if any(IDS[k] in blocks for k in pos):
            row['state']='held_corporate_event_review'
            continue
        r=h['risk'];last=h['last']
        if not timedelta(0)<=now-moment(last['at'])<=timedelta(days=4):
            row['state']='risk_history_stale'
            continue
        week=now.date().isocalendar()
        week=f'{week.year}-{week.week:02d}'
        day_start=number(r['day_equity'] if r['day']==now.date().isoformat() else last['equity'])
        week_start=number(r['week_equity'] if r['week']==week else last['equity'])
        peak=max(number(r['peak']),eq)
        if eq<=0 or eq<=day_start*D('.98') or eq<=week_start*D('.95') or eq<=peak*D('.90'):
            row['state']='risk_blocked'
            continue
        # Do not churn held names that remain in the target. Propose only
        # complete exits from the target and new names, not constant resizing.
        turn=cost=D(0)
        changes=[]
        for k in sorted(set(pos)-set(budgets[a])):
            bid,ask,bs,az=quotes[k]
            qty=pos[k]
            gross=qty*bid
            if qty>bs or gross>adv.get(k,D(0))*D('.001') or gross+turn>eq*MAX_TURNOVER:
                continue
            proceeds=qty*bid*(1-SLIP)
            charge=fee(proceeds)
            if proceeds<=charge:continue
            cash+=proceeds-charge
            cost+=gross-proceeds+charge
            turn+=gross
            del pos[k]
            changes.append({'ticker':IDS[k],'side':'sell','quantity':str(qty),'estimated_fee_pln':str(charge)})
        # New names in ranking order. Scale target weights to current equity.
        for k,budget in budgets[a].items():
            if k in pos or IDS[k] in blocks:continue
            bid,ask,bs,az=quotes[k]
            price=ask*(1+SLIP)
            equity=cash+value(pos)
            invested=value(pos)
            sector=SECTORS[IDS[k]]
            sector_value=sum((q*quotes[j][0] for j,q in pos.items() if SECTORS[IDS[j]]==sector),D(0))
            # Budget is deliberately conservative, based on all-in ask cost.
            cap=min(budget*equity/D(50000),equity*D('.045'),
                    equity*EXPOSURE[p]-invested,equity*D('.20')-sector_value,
                    cash-equity*D('.20'),eq*MAX_TURNOVER-turn,adv[k]*D('.001'))
            if cap<equity*MIN_WEIGHT:continue
            qty=min(az,max(D(0),((cap-MIN_FEE-CENT)/(price*(1+RATE))).to_integral_value(rounding=ROUND_DOWN)))
            if qty<1 or qty*bid<equity*MIN_WEIGHT:continue
            charge=fee(qty*price)
            debit=qty*price+charge
            after_cash=cash-debit
            after_pos=dict(pos,**{k:qty})
            after_invested=value(after_pos)
            after=after_cash+after_invested
            if (debit>cap or after_cash<after*D('.20') or after_invested>after*EXPOSURE[p]
                or qty*bid>after*D('.05') or sector_value+qty*bid>after*D('.20')
                or after<=day_start*D('.98') or after<=week_start*D('.95') or after<=peak*D('.90')):
                continue
            cash=after_cash;pos=after_pos
            turn+=debit
            cost+=debit-qty*bid
            changes.append({'ticker':IDS[k],'side':'buy','quantity':str(qty),'estimated_fee_pln':str(charge)})
        row.update(state='proposal_only' if changes else 'no_change',proposed_changes=changes,
                   estimated_cost_pln=str(cost),turnover_pln=str(turn),
                   projected_cash_pln=str(cash),projected_equity_pln=str(cash+value(pos)))
    return dict(FLAGS,state='rebalance_preview',checked_utc=now.isoformat(),
                cost_model='IEX bid/ask plus 10bps slippage and 10bps commission min PLN2',
                cooldown_days=7,turnover_limit_pct=15,minimum_new_position_pct=1,
                corporate_blocks=blocks,accounts=rows)

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ledger',required=True,type=Path)
    p.add_argument('--audit',required=True,type=Path)
    a=p.parse_args()
    try:
        if not a.ledger.is_file():
            print(json.dumps(dict(FLAGS,state='waiting_for_initial_open')))
            return
        receipt,h=read_state(a.ledger)
        now=datetime.now(timezone.utc)
        if now-moment(receipt['generated_utc'])<COOLDOWN:
            print(json.dumps(dict(FLAGS,state='cooldown',earliest_review_utc=(moment(receipt['generated_utc'])+COOLDOWN).isoformat())))
            return
        audit=read_audit(a.audit,now)
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        today=now.astimezone(NY).date()
        feed=scan(corporate_client(key,secret),SYMBOLS,today-timedelta(days=60),today+timedelta(days=30))
        blocks=event_blocks(feed,audit,moment(receipt['generated_utc']).astimezone(NY).date(),today)
        plan=initial_preview(asset_client(key,secret),calendar_client(key,secret),bars_client(key,secret))
        snapshot=qualify(*clients(key,secret),include_quotes=True)
        if not snapshot['ready']:
            print(json.dumps(dict(FLAGS,state=snapshot['state'])))
            return
        result=propose(receipt,h,plan,snapshot,blocks,datetime.now(timezone.utc))
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps(dict(FLAGS,state='failed',error_type=type(exc).__name__)))
        raise SystemExit(1)
if __name__=='__main__':
    main()
