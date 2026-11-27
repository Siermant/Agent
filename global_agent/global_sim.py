"""Synthetic Global portfolio execution laboratory, isolated from the running agent.

Replay explicitly hypothetical targets against artificial bid/ask snapshots.
Not a historical backtest, not a live paper feed, and never a broker order.
"""
import argparse
import json
import re
import tempfile
from datetime import timedelta
from decimal import Decimal as D, ROUND_DOWN, ROUND_HALF_UP
from pathlib import Path
from .equity_data import moment
from .ledger import Ledger, CAPITALS, EXPOSURE, positive

ACCOUNTS = tuple(p+':global' for p in 'ABCDE')
CENT = D('.01')


def cost_number(value, high):
    v=D(str(value))
    if not v.is_finite() or not D(0)<=v<=high:
        raise ValueError('Invalid scenario costs')
    return v


def parse_frame(frame, previous):
    at=moment(frame['at'])
    generated=moment(frame['targets_generated_at'])
    if (previous is not None and at<=previous) or not timedelta(0)<at-generated<=timedelta(days=7):
        raise ValueError('Invalid frame or target chronology')
    if frame.get('synthetic_quotes') is not True:
        raise ValueError('Synthetic quotes required')
    quotes={}
    for sym,q in frame['quotes'].items():
        if not re.fullmatch(r'fixture:[A-Za-z0-9_.-]{1,64}',sym):
            raise ValueError('Fixture-only instrument identifier required')
        bid,ask=positive(q['bid']),positive(q['ask'])
        ts=moment(q['observed_at'])
        if bid>ask or not timedelta(0)<=at-ts<=timedelta(seconds=60):
            raise ValueError('Invalid or stale synthetic quote')
        quotes[sym]=(bid,ask,ts)
    targets=frame['targets']
    if set(targets)!=set(ACCOUNTS):
        raise ValueError('Expected all five Global accounts')
    parsed={}
    for account,items in targets.items():
        p=account[0]
        maximum=CAPITALS['global']*(EXPOSURE[p] if p!='E' else D(1))
        one=CAPITALS['global']*(D('.045') if p!='E' else D(1))
        if p=='E' and len(items)>1:
            raise ValueError('Benchmark must use one instrument')
        vals={s:cost_number(v,one) for s,v in items.items()}
        if not set(vals)<=set(quotes) or sum(vals.values(),D(0))>maximum:
            raise ValueError('Target outside fixture quotes or capital limit')
        parsed[account]=vals
    return at,quotes,parsed


def whole(amount,price):
    return max(D(0),(amount/price).to_integral_value(rounding=ROUND_DOWN))


def run(scenario):
    if scenario.get('synthetic_only') is not True or scenario.get('real_trading') is not False:
        raise ValueError('Explicit synthetic-only scenario required')
    fee=cost_number(scenario['commission_bps'],D(1000))/10000
    slip=cost_number(scenario['slippage_bps'],D(1000))/10000
    minimum=cost_number(scenario['minimum_fee_pln'],D(1000))
    frames=scenario['frames']
    if not 1<=len(frames)<=370:
        raise ValueError('Invalid scenario frame count')
    # Validate the complete scenario before doing any simulated fills.
    prepared=[]
    previous=None
    for frame in frames:
        parsed=parse_frame(frame,previous)
        previous=parsed[0]
        prepared.append(parsed)
    clock=[prepared[0][0]]
    executions, history, blocked=[],[],[]
    with tempfile.TemporaryDirectory(prefix='gta-global-sim-') as tmp:
        ledger=Ledger(Path(tmp)/'scenario.db',clock=lambda:clock[0])
        def state(account):
            return next(s for s in ledger.status() if s['account']==account)
        def fill(account,sym,side,qty,quotes,index):
            if qty<=0:
                return
            try:
                result=ledger.execute(account,f'fixture:{index}:{sym}:{side}',sym,side,qty,
                                      quotes,fee,slip,minimum,market_open=True,approved=True)
            except ValueError as exc:
                if str(exc) not in ('Risk limit','Insufficient cash','Insufficient cash for fees'):
                    raise
                blocked.append({'frame':index,'account':account,'symbol':sym,'side':side,'reason':str(exc)})
            else:
                executions.append({'frame':index,'account':account,'symbol':sym,'side':side,
                                   'quantity':str(qty),'price_pln':result['price_pln'],'fee_pln':result['fee_pln']})
        for index,(at,quotes,targets) in enumerate(prepared):
            clock[0]=at
            for account in ACCOUNTS:
                s=state(account)
                if not set(s['positions'])<=set(quotes):
                    raise ValueError('Missing quote for held position')
            for account in ACCOUNTS:
                s=state(account)
                ledger.mark(account,quotes)
                # Benchmark is a one-time buy; later frames only value it.
                if account.startswith('E:') and s['fills']:
                    continue
                desired=targets[account]
                # Size targets using the bid for reductions; whole shares only.
                for sym,qty in sorted(s['positions'].items()):
                    keep=whole(desired.get(sym,D(0)),quotes[sym][0])
                    if qty>keep:
                        fill(account,sym,'sell',qty-keep,quotes,index)
                for sym,budget in sorted(desired.items()):
                    s=state(account)
                    held=s['positions'].get(sym,D(0))
                    price=quotes[sym][1]*(1+slip)
                    # Target budgets include costs and current position value.
                    available=min(budget-held*quotes[sym][0],D(s['cash_pln']))
                    qty=whole(max(D(0),available-minimum-CENT),price*(1+fee))
                    if qty>0:
                        value=qty*price
                        commission=max(minimum,(value*fee).quantize(CENT,rounding=ROUND_HALF_UP))
                        if value+commission<=available:
                            fill(account,sym,'buy',qty,quotes,index)
                ledger.mark(account,quotes)
            history.append({'at':at.isoformat(),'accounts':[
                {'account':s['account'],'cash_pln':s['cash_pln'],'equity_pln':s['last_equity_pln'],'fills':s['fills']}
                for s in ledger.status() if s['account'] in ACCOUNTS]})
        final=[dict(s,positions={k:str(v) for k,v in s['positions'].items()})
               for s in ledger.status() if s['account'] in ACCOUNTS]
    return {'synthetic_only':True,'research_only':True,'real_trading':False,
            'eligible_for_orders':False,'isolated_temporary_ledger':True,
            'history':history,'executions':executions,'blocked':blocked,'accounts':final}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(run(json.loads(a.scenario.read_text(encoding='utf-8'))),indent=2,allow_nan=False))

if __name__=='__main__':
    main()
