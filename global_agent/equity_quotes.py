"""Read-only qualification of intraday US equity quotes and reference FX.
Never sends orders. A valid quote is only an IEX simulation input, not an
executable consolidated quote. Missing/stale instruments block readiness.
"""
import json
import os
from datetime import datetime,timedelta,timezone
from decimal import Decimal as D
from urllib.parse import urlencode
from urllib.request import Request,build_opener,ProxyHandler
from urllib.error import HTTPError
from .alpaca_data import NoRedirect,NY
from .equity_data import moment,positive
from .equity_job import credentials,QUALIFIED
from .fx_reference import rates

SYMBOLS=sorted(s for ss in QUALIFIED.values() for s in ss if s not in ('QQQ','TLT'))
MAX_AGE=timedelta(seconds=60)
MAX_SPREAD=D('.01')


def clients(key,secret):
    if not key or not secret or any(c.isspace() for c in key+secret):
        raise ValueError('Invalid credentials')
    opener=build_opener(ProxyHandler({}),NoRedirect())
    def read(url):
        req=Request(url,headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
                                 'Accept':'application/json','User-Agent':'GTA-research'})
        with opener.open(req,timeout=15) as response:
            raw=response.read(1000001)
        if len(raw)>1000000:
            raise ValueError('Oversized market response')
        return json.loads(raw,parse_float=D)
    return (lambda:read('https://paper-api.alpaca.markets/v2/clock'),
            lambda:read('https://data.alpaca.markets/v2/stocks/quotes/latest?'+
                        urlencode({'symbols':','.join(SYMBOLS),'feed':'iex','currency':'USD'})))


def market_clock(payload,now):
    stamp=moment(payload['timestamp'])
    opened=payload['is_open']
    nxt_open=moment(payload['next_open'])
    nxt_close=moment(payload['next_close'])
    if type(opened) is not bool or abs(now-stamp)>timedelta(seconds=15):
        raise ValueError('Invalid or stale market clock')
    if opened and not now<nxt_close:
        raise ValueError('Market closing or inconsistent clock')
    if not opened and nxt_open<=now:
        raise ValueError('Invalid next session')
    return opened,nxt_close,nxt_open


def qualify(clock_request,quote_request,fx_request=None,clock=None):
    clock=clock or (lambda:datetime.now(timezone.utc))
    first=clock_request()
    now=clock()
    opened,close,nxt=market_clock(first,now)
    result={'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'source':'alpaca_iex_latest','fx_source':'nbp_table_a_prior_day',
            'executable_fx_verified':False,'consolidated_quotes_verified':False,
            'expected_instruments':len(SYMBOLS),'ready':False,'checked_utc':now.isoformat(),
            'market_open':opened,'next_open':nxt.isoformat()}
    if not opened:
        return dict(result,state='market_closed')
    # FX first, then obtain quotes so their lifetime is not spent waiting for NBP.
    day=now.astimezone(NY).date()
    kwargs={} if fx_request is None else {'request':fx_request}
    fx=rates('USD',day-timedelta(days=7),day-timedelta(days=1),**kwargs)
    if not fx:
        raise ValueError('Missing prior USD FX')
    fx_day=max(fx)
    payload=quote_request()
    second=clock_request()
    now=clock()
    opened2,close2,_=market_clock(second,now)
    result['checked_utc']=now.isoformat()
    if not opened2 or close2!=close or now.astimezone(NY).date()!=day:
        return dict(result,state='session_changed',market_open=opened2)
    quotes=payload['quotes']
    if not isinstance(quotes,dict) or not set(quotes)<=set(SYMBOLS):
        raise ValueError('Unexpected quote universe')
    if payload.get('currency','USD')!='USD':
        raise ValueError('Unexpected currency')
    accepted={}
    rejected=[]
    for s in SYMBOLS:
        try:
            q=quotes[s]
            at=moment(q['t'])
            bid,ask=positive(str(q['bp'])),positive(str(q['ap']))
            bs,az=positive(str(q['bs'])),positive(str(q['as']))
            if not timedelta(0)<=now-at<=MAX_AGE or at.astimezone(NY).date()!=day:
                raise ValueError('Stale quote')
            if bid>ask or (ask-bid)/bid>MAX_SPREAD:
                raise ValueError('Crossed or wide quote')
            accepted[s]={'bid_pln_reference':str(bid*fx[fx_day]),
                         'ask_pln_reference':str(ask*fx[fx_day]),'observed_at':at.isoformat()}
        except (KeyError,ValueError,TypeError,ArithmeticError):
            rejected.append(s)
    return dict(result,state='qualified' if not rejected else 'incomplete_quotes',
                ready=not rejected,fx_date=str(fx_day),valid_quotes=len(accepted),
                rejected_symbols=rejected,session_close_utc=close.isoformat())


def main():
    try:
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        r=qualify(*clients(key,secret))
        print(json.dumps(r,indent=2,allow_nan=False))
    except Exception as exc:
        error={'state':'failed','error_type':type(exc).__name__}
        if isinstance(exc,HTTPError):
            error['http_status']=exc.code
        print(json.dumps(error))
        raise SystemExit(1)
    if not r['ready'] and r['state']!='market_closed':
        raise SystemExit(2)

if __name__=='__main__':
    main()
