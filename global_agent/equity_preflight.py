"""Read-only US equity data qualification with paper-only credentials.

No order calls, persistent credentials, ledger writes or published market data.
"""
import argparse
import getpass
import json
import sys
import warnings
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, ProxyHandler
from .alpaca_data import client, history, NoRedirect, NY
from .equity_data import moment
from .fx_reference import rates

VENUES={'XNAS':('AAPL','MSFT','QQQ'),'XNYS':('IBM',)}
BASE='https://paper-api.alpaca.markets/v3/calendar/'


def calendar_client(key,secret):
    if not key or not secret or any(c.isspace() for c in key+secret):
        raise ValueError('Missing paper credentials')
    opener=build_opener(ProxyHandler({}),NoRedirect())
    def request(mic,start,end):
        if mic not in VENUES:
            raise ValueError('Unsupported venue')
        url=BASE+mic+'?'+urlencode({'start':str(start),'end':str(end),'timezone':'UTC'})
        req=Request(url,headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
                                 'Accept':'application/json','User-Agent':'GTA-research'})
        with opener.open(req,timeout=20) as response:
            raw=response.read(1_000_001)
        if len(raw)>1_000_000:
            raise ValueError('Oversized calendar response')
        return json.loads(raw)
    return request


def parse_calendar(payload,mic,start,end,now):
    if payload.get('market',{}).get('mic')!=mic or not isinstance(payload.get('calendar'),list):
        raise ValueError('Unexpected calendar response')
    sessions={}
    for row in payload['calendar']:
        d=date.fromisoformat(row['date'])
        op,cl=moment(row['core_start']),moment(row['core_end'])
        if (not start<=d<=end or d in sessions or
                op.astimezone(NY).date()!=d or cl.astimezone(NY).date()!=d or
                not op<cl<=now):
            raise ValueError('Invalid calendar session')
        sessions[d]=cl
    if len(sessions)<61 or (end-max(sessions)).days>5:
        raise ValueError('Insufficient completed sessions')
    return sessions


def qualify(calendar_request,bar_request,fx_request=None,now=None):
    now=now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError('Timezone required')
    end=now.astimezone(NY).date()-timedelta(days=1)
    start=end-timedelta(days=119)
    result={'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'checked_utc':now.isoformat(),'source':'alpaca_iex_paper_calendar',
            'checks':{},'independent_exchange_calendar_verified':False}
    def check(name,fn):
        try:
            result['checks'][name]={'ok':True,**fn()}
        except Exception as exc:
            error={'ok':False,'error_type':type(exc).__name__}
            if isinstance(exc,HTTPError):
                error['http_status']=exc.code
            result['checks'][name]=error
    def fx():
        kwargs={} if fx_request is None else {'request':fx_request}
        counts={}
        for cur in ('USD','EUR'):
            data=rates(cur,start-timedelta(days=10),end,**kwargs)
            ordered=sorted(data)
            if not ordered or not 0<=(end-ordered[-1]).days<=7:
                raise ValueError('Missing recent FX')
            if not any(0<(start-d).days<=7 for d in ordered):
                raise ValueError('Missing prior FX')
            counts[cur]=len(data)
        return {'rate_counts':counts}
    check('nbp',fx)
    for mic,symbols in VENUES.items():
        def venue(mic=mic,symbols=symbols):
            sessions=parse_calendar(calendar_request(mic,start,end),mic,start,end,now)
            expected={(s,d) for s in symbols for d in sessions}
            raw=history(bar_request,expected,'raw')
            adjusted=history(bar_request,expected,'split,dividend')
            if set(raw)!=expected or set(adjusted)!=expected:
                raise ValueError('Incomplete market history')
            return {'sessions':len(sessions),'symbols':list(symbols),
                    'raw_bars':len(raw),'adjusted_bars':len(adjusted),
                    'last_session':str(max(sessions))}
        check(mic,venue)
    result['ready']=all(x['ok'] for x in result['checks'].values())
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',action='store_true',required=True,
                   help='Enter paper-only credentials locally, without echo')
    p.parse_args()
    if not sys.stdin.isatty():
        p.error('Interactive terminal required; credentials must not be piped')
    warnings.filterwarnings('error',category=getpass.GetPassWarning)
    try:
        key=getpass.getpass('Alpaca PAPER API key: ')
        secret=getpass.getpass('Alpaca PAPER secret: ')
        result=qualify(calendar_client(key,secret),client(key,secret))
    except Exception as exc:
        p.exit(1,'Preflight failed: '+type(exc).__name__+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    if not result['ready']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
