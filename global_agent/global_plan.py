"""Research-only initial allocation from one comparable equity ranking cohort.

No existing holdings, order instructions, broker API, or ledger writes.
Costs and liquidity are explicitly supplied assumptions, not verified quotes.
"""
import argparse
import json
from decimal import Decimal as D, ROUND_DOWN
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .equity_data import moment
from .ledger import CAPITALS, EXPOSURE

CENT = D('.01')
POSITION_BUDGET = D('.045')
SECTOR_CAP = D('.20')
MIN_CASH = D('.20')


def number(value, low, high):
    v = D(str(value))
    if not v.is_finite() or not low <= v <= high:
        raise ValueError('Invalid numeric input')
    return v


def plan(ranking, assumptions, cohort_index=0, now=None, *, aligned_sessions=None):
    now = now or datetime.now(timezone.utc)
    if (ranking.get('research_only') is not True or ranking.get('real_trading') is not False
            or ranking.get('eligible_for_orders') is not False or ranking.get('currency') != 'PLN'):
        raise ValueError('Expected research ranking in PLN')
    generated = moment(ranking['generated_utc'])
    if not timedelta(0) <= now-generated <= timedelta(days=1):
        raise ValueError('Ranking expired or future dated')
    if assumptions.get('research_only') is not True:
        raise ValueError('Explicit research assumptions required')
    if cohort_index is None:
        cohorts=ranking['cohorts']
        if not aligned_sessions or len(cohorts)!=2 or {c['mic'] for c in cohorts}!={'XNYS','XNAS'}:
            raise ValueError('Explicit aligned US calendars required')
        if set(aligned_sessions)!={'XNYS','XNAS'} or aligned_sessions['XNYS']!=aligned_sessions['XNAS']:
            raise ValueError('Different US session calendars')
        dates=[d for d,_ in aligned_sessions['XNYS']]
        if len(dates)!=61 or dates!=sorted(set(dates)):
            raise ValueError('Invalid aligned session window')
        if any(c['source']!='alpaca_iex_daily' or c['sessions']!=61
               or c['first_session']!=dates[0] or c['last_session']!=dates[-1] for c in cohorts):
            raise ValueError('Ranking does not match aligned calendars')
    else:
        if not 0 <= cohort_index < len(ranking['cohorts']):
            raise ValueError('Invalid cohort')
        cohorts=[ranking['cohorts'][cohort_index]]
    pairs=[(c,r) for c in cohorts for r in c['instruments']]
    rows=[r for _,r in pairs]
    if len({r['instrument_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ranked instrument')
    meta = assumptions['instruments']
    fee_bps = number(assumptions['commission_bps'], D(0), D(1000))
    impact_bps = number(assumptions['spread_slippage_bps'], D(0), D(1000))
    minimum_fee = number(assumptions['minimum_fee_pln'], D(0), D(1000))
    minimum_adv = number(assumptions['minimum_adv_pln'], D(1), D('1e15'))
    max_adv_fraction = number(assumptions['max_adv_fraction'], D('.000001'), D('.01'))
    capital = CAPITALS['global']
    candidates, excluded = [], []
    for cohort,r in pairs:
        ident = r['instrument_id']
        if r['mic'] != cohort['mic'] or r['source'] != cohort['source'] or r['last_session'] != cohort['last_session']:
            raise ValueError('Inconsistent cohort')
        m = meta.get(ident)
        if not m or m.get('research_eligible') is not True or not isinstance(m.get('sector'),str) or not m['sector'].strip():
            excluded.append({'instrument_id':ident,'reason':'missing_research_metadata'})
            continue
        adv = number(m['adv_pln'],D(0),D('1e15'))
        if adv < minimum_adv:
            excluded.append({'instrument_id':ident,'reason':'assumed_low_liquidity'})
            continue
        v = {k:number(r[k],D('-1000000'),D('1000000')) for k in ('return_20_pct','return_60_pct','annualized_volatility_pct','max_drawdown_60_pct','score')}
        if v['annualized_volatility_pct']<0 or not 0<=v['max_drawdown_60_pct']<=100:
            raise ValueError('Invalid risk metrics')
        candidates.append((r,m,adv,v))

    def score(p,v):
        a,b,vol,dd=(v[k] for k in ('return_20_pct','return_60_pct','annualized_volatility_pct','max_drawdown_60_pct'))
        if p=='A':
            return D('.3')*a+D('.7')*b-D('.6')*vol-dd
        if p=='B':
            return v['score']
        if p=='C':
            return D('.7')*a+D('.3')*b-D('.15')*vol-D('.25')*dd
        return D('.4')*a+D('.6')*b-D('.4')*vol-D('.7')*dd

    plans=[]
    for p in 'ABCD':
        cash=capital
        invested=costs=D(0)
        sectors={}
        allocations=[]
        for r,m,adv,v in sorted(candidates,key=lambda x:(-score(p,x[3]),x[0]['instrument_id'])):
            if score(p,v)<=0:
                continue
            sector=m['sector']
            budget=min(capital*POSITION_BUDGET, capital*EXPOSURE[p]-invested,
                       capital*SECTOR_CAP-sectors.get(sector,D(0)),adv*max_adv_fraction,
                       cash-capital*MIN_CASH)
            # Budget includes assumed entry costs; round principal down.
            if budget<=minimum_fee:
                continue
            principal=min((budget-minimum_fee)/(1+(fee_bps+impact_bps)/10000),
                          budget/(1+(fee_bps+impact_bps)/10000)).quantize(CENT,rounding=ROUND_DOWN)
            if principal<=0:
                continue
            cost=max(minimum_fee,principal*fee_bps/10000)+principal*impact_bps/10000
            cost=cost.quantize(CENT,rounding='ROUND_UP')
            if principal+cost>budget:
                continue
            cash-=principal+cost
            invested+=principal
            costs+=cost
            sectors[sector]=sectors.get(sector,D(0))+principal
            allocations.append({'instrument_id':r['instrument_id'],'sector':sector,
                                'notional_pln':str(principal),'estimated_entry_cost_pln':str(cost)})
        plans.append({'account':p+':global','capital_pln':str(capital),'cash_pln':str(cash),
                      'allocated_pln':str(invested),'estimated_cost_pln':str(costs),'allocations':allocations})
    # Benchmark is explicitly selected, never the highest-ranked stock.
    bench=assumptions.get('benchmark_id')
    match=next((x for x in candidates if x[0]['instrument_id']==bench and x[0]['kind']=='etf' and x[1].get('benchmark_eligible') is True),None)
    allocations=[]
    principal=cost=D(0)
    if match:
        r,m,adv,_=match
        budget=min(capital,adv*max_adv_fraction)
        if budget>minimum_fee:
            principal=((budget-minimum_fee)/(1+(fee_bps+impact_bps)/10000)).quantize(CENT,rounding=ROUND_DOWN)
            cost=(max(minimum_fee,principal*fee_bps/10000)+principal*impact_bps/10000).quantize(CENT,rounding='ROUND_UP')
            if principal>0 and principal+cost<=budget:
                allocations=[{'instrument_id':bench,'sector':m['sector'],'notional_pln':str(principal),'estimated_entry_cost_pln':str(cost)}]
    if not allocations:
        principal=cost=D(0)
    plans.append({'account':'E:global','capital_pln':str(capital),'cash_pln':str(capital-principal-cost),
                  'allocated_pln':str(principal),'estimated_cost_pln':str(cost),'allocations':allocations})
    return {'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'initial_cash_only':True,'assumptions_verified':False,'etf_lookthrough':False,
            'method':'initial_allocation_v1','cohort_index':cohort_index,'aligned_us_research':cohort_index is None,'plans':plans,'excluded':excluded}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ranking',type=Path,required=True)
    p.add_argument('--assumptions',type=Path,required=True)
    p.add_argument('--cohort',type=int,default=0)
    a=p.parse_args()
    print(json.dumps(plan(json.loads(a.ranking.read_text()),json.loads(a.assumptions.read_text()),a.cohort),indent=2,allow_nan=False))

if __name__=='__main__':
    main()
