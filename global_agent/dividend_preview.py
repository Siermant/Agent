"""Read-only dividend entitlement preview for the one-time Global experiment.
No cash posting, no tax assumptions, no FX conversion, no orders.
Missing currency may be used only in a separately labelled USD scenario.
"""
import json
import os
from datetime import datetime,timedelta,date,timezone
from decimal import Decimal as D
from pathlib import Path
from .alpaca_data import NY
from .equity_data import moment,positive
from .equity_job import credentials
from .global_paper_initial import ACCOUNTS,IDS,read_existing
from .global_paper_mark import read_holdings
from .corporate_actions import client,scan


def preview(receipt,report,now=None):
    now=now or datetime.now(timezone.utc)
    if (receipt.get('state') not in ('opened','already_opened')
        or receipt.get('real_trading') is not False or receipt.get('simulated') is not True
        or receipt.get('one_time_initial_only') is not True or receipt.get('rebalance_enabled') is not False):
        raise ValueError('Not the initial-only paper experiment')
    if (report.get('state')!='feed_qualified' or report.get('real_trading') is not False
        or report.get('accounting_enabled') is not False or report.get('coverage_guaranteed') is not False):
        raise ValueError('Unexpected corporate action report')
    today=now.astimezone(NY).date()
    holdings={}
    for f in receipt['fills']:
        a,k=f['account'],f['instrument_id']
        if a not in ACCOUNTS or a=='E:global' or k not in IDS or (a,k) in holdings or f.get('simulated') is not True:
            raise ValueError('Unexpected initial fill')
        # Opening fills have no timestamp of their own in the receipt. Use the
        # later receipt timestamp conservatively; never credit same-day ex-date.
        opened=moment(receipt['generated_utc']).astimezone(NY).date()
        qty=positive(str(f['quantity']))
        if opened>today or qty!=qty.to_integral_value():
            raise ValueError('Invalid opening')
        holdings[a,k]=(qty,opened)
    accounts={r['account']:r for r in receipt['accounts']}
    if len(receipt['accounts'])!=5 or set(accounts)!=set(ACCOUNTS):
        raise ValueError('Missing accounts')
    for a,r in accounts.items():
        expected={k:qty for (account,k),(qty,_) in holdings.items() if account==a}
        if {k:D(str(q)) for k,q in r['positions'].items()}!=expected:
            raise ValueError('Opening positions mismatch')
    events=report['events']
    if len(events)!=report['events_count'] or len({e['id'] for e in events})!=len(events):
        raise ValueError('Inconsistent corporate action feed')
    # Any split, merger or other non-cash event on held symbol can invalidate
    # initial quantities; block forecasts for that symbol rather than guessing.
    blocked={s for e in events if e['type']!='cash_dividends' for s in e['symbols']}
    clusters={}
    for e in events:
        if e['type']=='cash_dividends' and e['dates'].get('ex_date'):
            for s in e['symbols']:
                clusters.setdefault((s,e['dates']['ex_date']),[]).append(e['id'])
    overlaps={k for k,ids in clusters.items() if len(ids)>1}
    rows=[]
    totals={a:{'explicit_usd_gross':'0','assumed_usd_gross':'0'} for a in ACCOUNTS}
    for e in events:
        if e['type']!='cash_dividends':
            continue
        for (a,k),(qty,opened) in holdings.items():
            s=IDS[k]
            if s not in e['symbols']:
                continue
            dates=e['dates']
            r={'account':a,'ticker':s,'event_id':e['id'],'dates':dates,'quantity':str(qty),
               'state':'manual_review_required'}
            rows.append(r)
            if 'ex_date' not in dates:
                continue
            ex=date.fromisoformat(dates['ex_date'])
            if ex<=opened:
                r['state']='not_entitled_opened_on_or_after_ex_date'
                continue
            if (s,dates['ex_date']) in overlaps:
                r['state']='overlapping_events_review_required'
                continue
            if s in blocked or 'payable_date' not in dates or len(e['symbols'])!=1:
                continue
            if date.fromisoformat(dates['payable_date'])<ex:
                continue
            explicit=e.get('state')=='simple_dividend_candidate' and e.get('currency_status')=='USD'
            assumed=(e.get('review_reasons')==['currency_not_explicit_usd']
                     and e.get('currency_status') in ('missing','empty')
                     and e.get('foreign_status')=='false' and e.get('special_status')=='false'
                     and e.get('rate_valid') is True)
            if not (explicit or assumed):
                continue
            rate=positive(str(e['rate_usd'] if explicit else e['reported_rate']))
            if rate>D(10000):
                raise ValueError('Invalid rate')
            gross=qty*rate
            field='explicit_usd_gross' if explicit else 'assumed_usd_gross'
            r.update(state='forecast' if ex>today else 'potential_entitlement_unreconciled',
                     currency_basis='provider_explicit_usd' if explicit else 'assumption_only_usd',
                     gross_usd=str(gross))
            # Totals include potential entitlements and future forecasts, not paid cash.
            totals[a][field]=str(D(totals[a][field])+gross)
    return {'state':'entitlement_preview','research_only':True,'real_trading':False,
            'accounting_enabled':False,'eligible_for_orders':False,'coverage_guaranteed':False,
            'gross_before_tax':True,'totals_are_not_cash':True,'checked_utc':now.isoformat(),
            'blocked_symbols':sorted(blocked),'overlapping_cash_events':len(overlaps),'accounts':totals,'events':rows}


def main():
    try:
        root=Path(os.environ['STATE_DIRECTORY'])
        if root.name!='gta-global-paper':
            raise ValueError('Unexpected isolated state directory')
        path=root/'global-paper.db'
        if not path.is_file():
            print(json.dumps({'state':'waiting_for_initial_open','real_trading':False,'accounting_enabled':False}))
            return
        receipt=read_existing(path)
        h=read_holdings(path)
        selected=sorted({IDS[k] for r in h.values() for k in r['positions']})
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        now=datetime.now(timezone.utc)
        today=now.astimezone(NY).date()
        result=preview(receipt,scan(client(key,secret),selected,today-timedelta(days=60),today+timedelta(days=30)),now)
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'state':'failed','error_type':type(exc).__name__,'real_trading':False,'accounting_enabled':False}))
        raise SystemExit(1)

if __name__=='__main__':
    main()
