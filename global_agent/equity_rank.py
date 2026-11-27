"""Transparent retrospective equity/ETF screening, never an order signal."""
import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import stdev

from .equity_data import moment, positive

WINDOW = 61
MAX_AGE_DAYS = 5


def metrics(prices):
    if len(prices) != WINDOW or any(not math.isfinite(p) or p <= 0 for p in prices):
        raise ValueError('Invalid price window')
    returns = [b/a-1 for a,b in zip(prices,prices[1:])]
    ret20 = 100*(prices[-1]/prices[-21]-1)
    ret60 = 100*(prices[-1]/prices[0]-1)
    vol = 100*stdev(returns)*math.sqrt(252)
    peak = prices[0]
    drawdown = 0.0
    for p in prices:
        peak = max(peak,p)
        drawdown = max(drawdown,100*(1-p/peak))
    # Research heuristic in percentage points, not a forecast or optimized weights.
    score = .4*ret20 + .6*ret60 - .25*vol - .5*drawdown
    if not all(math.isfinite(v) for v in (ret20,ret60,vol,drawdown,score)):
        raise ValueError('Nonfinite metric')
    return {'return_20_pct':ret20,'return_60_pct':ret60,
            'annualized_volatility_pct':vol,'max_drawdown_60_pct':drawdown,
            'score':score}


def rank(dataset, now=None):
    now = now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError('Timezone required')
    if dataset.get('research_only') is not True or dataset.get('real_trading') is not False:
        raise ValueError('Expected research-only dataset')
    grouped = defaultdict(list)
    seen = set()
    for row in dataset['bars']:
        ident = row['instrument_id']
        day = date.fromisoformat(row['session'])
        close = moment(row['close_at'])
        retrieved = moment(row['retrieved_at'])
        key = ident,day
        if key in seen or not close <= retrieved <= now or day > now.date():
            raise ValueError('Duplicate or future data')
        seen.add(key)
        grouped[ident].append((day,row))
    if not grouped or len(grouped) != dataset.get('instruments'):
        raise ValueError('Empty or inconsistent instrument set')
    cohorts = defaultdict(list)
    excluded = []
    for ident, series in sorted(grouped.items()):
        series.sort(key=lambda x:x[0])
        latest = series[-1][1]
        base = {'instrument_id':ident,'ticker':latest['ticker'],
                'mic':latest['mic'],'kind':latest['kind'],
                'source':latest['source'],'last_session':str(series[-1][0])}
        signature = ('ticker','mic','kind','source','currency','adjustment')
        if any(tuple(row[k] for k in signature)!=tuple(latest[k] for k in signature) for _,row in series):
            raise ValueError('Inconsistent instrument metadata')
        reason = None
        if (now.date()-series[-1][0]).days > MAX_AGE_DAYS:
            reason='stale_history'
        elif len(series)<WINDOW:
            reason='insufficient_sessions'
        elif latest['adjustment']!='split_dividend' or any(row.get('adjusted_close_pln') is None for _,row in series[-WINDOW:]):
            reason='missing_total_return_adjustment'
        if reason:
            excluded.append(dict(base,reason=reason))
            continue
        window=series[-WINDOW:]
        dates=tuple(d.isoformat() for d,_ in window)
        prices=[float(positive(row['adjusted_close_pln'])) for _,row in window]
        # Compare only identical observation windows from the same venue/feed.
        cohorts[(latest['mic'],latest['source'],dates)].append(dict(base,**metrics(prices)))
    rankings=[]
    for (mic,source,dates), items in sorted(cohorts.items()):
        items.sort(key=lambda row:(-row['score'],row['instrument_id']))
        rankings.append({'mic':mic,'source':source,'first_session':dates[0],
                         'last_session':dates[-1],'sessions':WINDOW,
                         'instruments':[dict(row,rank=i+1) for i,row in enumerate(items)]})
    return {'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'generated_utc':now.isoformat(),'currency':'PLN',
            'calendar_verified_externally':False,
            'point_in_time_verified':False,'liquidity_evaluated':False,
            'costs_evaluated':False,'sector_diversification_evaluated':False,
            'method':'momentum_risk_v1','cohorts':rankings,'excluded':excluded}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    a=p.parse_args()
    with a.input.open(encoding='utf-8') as f:
        result=rank(json.load(f))
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
