"""Read-only initial allocation preview with observed IEX turnover proxy.
Not execution, not a backtest; scenario costs and sector labels are assumptions.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from urllib.error import HTTPError
from . import alpaca_data, fx_reference
from .equity_job import credentials, QUALIFIED, expanded_scan
from .equity_expand import asset_client
from .equity_preflight import calendar_client,parse_calendar
from .equity_scan import scan
from .global_plan import plan

GROUPS={
    'consumer_staples':'WMT COST PEP KO',
    'technology':'AAPL MSFT AMD NVDA AVGO IBM ORCL',
    'consumer_discretionary':'AMZN',
    'communication':'META GOOGL',
    'industrials':'HON GE CAT',
    'finance':'MA V JPM BAC',
    'healthcare':'ABBV JNJ UNH',
    'energy':'CVX XOM',
    'fund':'QQQ TLT',
}
SECTORS={s:k for k,v in GROUPS.items() for s in v.split()}


def turnover(bars, rates, sessions):
    if len(sessions)!=20 or sessions!=sorted(set(sessions)):
        raise ValueError('Expected 20 ordered sessions')
    dates=sorted(rates)
    total=D(0)
    for day in sessions:
        previous=[d for d in dates if d<day]
        if not previous or day-previous[-1]>timedelta(days=7):
            raise ValueError('Missing prior USD FX')
        _,volume,vwap=bars[day]
        total+=D(volume)*vwap*D(str(rates[previous[-1]]))
    return total/D(20)


def preview(assets, calendars, bars, fx_request=None, now=None):
    now=now or datetime.now(timezone.utc)
    end=now.astimezone(alpaca_data.NY).date()-timedelta(days=1)
    start=end-timedelta(days=119)
    # Reuse exactly the same calendar responses for rank and liquidity.
    cached={mic:calendars(mic,start,end) for mic in QUALIFIED}
    def calendar(mic,a,b):
        if (a,b)!=(start,end):
            raise ValueError('Calendar window changed')
        return cached[mic]
    def scanner(cal,req,universe):
        return scan(cal,req,fx_request,clock=lambda:now,universe=universe)
    ranking=expanded_scan(assets,calendar,bars,scanner)
    if ranking['state']!='ranked':
        raise ValueError('Incomplete ranking')
    parsed={m:sorted(parse_calendar(p,m,start,end,now).items())[-61:] for m,p in cached.items()}
    if parsed['XNAS']!=parsed['XNYS']:
        raise ValueError('Different venue sessions')
    sessions=[d for d,_ in parsed['XNAS']][-20:]
    expected={(s,d) for symbols in QUALIFIED.values() for s in symbols for d in sessions}
    volumes=alpaca_data.history(bars,expected,'raw',with_volume=True)
    kwargs={} if fx_request is None else {'request':fx_request}
    rates=fx_reference.rates('USD',sessions[0]-timedelta(days=10),sessions[-1],**kwargs)
    assumptions={'research_only':True,'commission_bps':'10','spread_slippage_bps':'10',
                 'minimum_fee_pln':'2','minimum_adv_pln':'100000','max_adv_fraction':'.001',
                 'benchmark_id':None,'instruments':{}}
    liquid=[]
    for cohort in ranking['cohorts']:
        for r in cohort['instruments']:
            s=r['ticker']
            adv=turnover({d:volumes[s,d] for d in sessions},rates,sessions)
            sector=SECTORS[s]
            eligible=r['kind']=='stock'
            assumptions['instruments'][r['instrument_id']]={'research_eligible':eligible,'sector':sector,'adv_pln':str(adv)}
            liquid.append({'ticker':s,'mic':r['mic'],'sector_assumed':sector,
                           'iex_turnover_20_pln':str(adv.quantize(D('.01'))),
                           'research_eligible':eligible and adv>=D(100000)})
    aligned={m:[(str(d),cl.isoformat()) for d,cl in rows] for m,rows in parsed.items()}
    result=plan(ranking,assumptions,None,now=max(now,datetime.now(timezone.utc)),aligned_sessions=aligned)
    return dict(result,generated_utc=ranking['generated_utc'],state='allocation_preview',
                liquidity_source='iex_raw_volume_times_vwap_prior_nbp_fx',
                liquidity_is_proxy=True,consolidated_market_liquidity_verified=False,
                benchmark_state='waiting_for_appropriate_broad_ucits_etf',
                instruments=liquid,cost_scenario={k:v for k,v in assumptions.items() if k not in ('instruments','benchmark_id')},
                calendar_verified_externally=False,point_in_time_verified=False)


def main():
    try:
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        result=preview(asset_client(key,secret),calendar_client(key,secret),alpaca_data.client(key,secret))
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        error={'state':'failed','error_type':type(exc).__name__}
        if isinstance(exc,HTTPError):
            error['http_status']=exc.code
        print(json.dumps(error))
        raise SystemExit(1)

if __name__=='__main__':
    main()
