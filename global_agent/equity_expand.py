"""One-time expansion qualification using Paper asset metadata, not orders."""
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
from urllib.parse import quote
from uuid import UUID
from .alpaca_data import NoRedirect, client
from .equity_preflight import calendar_client
from .equity_scan import scan
from .equity_job import credentials

STOCKS='AAPL MSFT NVDA AMD AVGO ORCL AMZN GOOGL META JPM BAC V MA XOM CVX JNJ ABBV UNH CAT GE HON KO PEP WMT COST IBM'.split()
ETFS='SPY QQQ VT TLT'.split()
CANDIDATES={**{s:'stock' for s in STOCKS},**{s:'etf' for s in ETFS}}
EXCHANGES={'NASDAQ':'XNAS','NYSE':'XNYS'}


def asset_client(key,secret):
    opener=build_opener(ProxyHandler({}),NoRedirect())
    def request(symbol):
        if symbol not in CANDIDATES:
            raise ValueError('Unknown candidate')
        url='https://paper-api.alpaca.markets/v2/assets/'+quote(symbol,safe='')
        req=Request(url,headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
                                 'Accept':'application/json','User-Agent':'GTA-research'})
        with opener.open(req,timeout=15) as r:
            raw=r.read(100001)
        if len(raw)>100000:
            raise ValueError('Oversized asset response')
        return json.loads(raw)
    return request


def discover(request):
    universe, excluded, identities = {}, [], set()
    for symbol,kind in CANDIDATES.items():
        try:
            a=request(symbol)
        except HTTPError as exc:
            if exc.code!=404:
                raise
            excluded.append({'ticker':symbol,'reason':'not_found'})
            continue
        if a.get('symbol')!=symbol or a.get('class')!='us_equity':
            raise ValueError('Unexpected asset response')
        ident=UUID(a['id'])
        if ident in identities:
            raise ValueError('Duplicate asset identifier')
        identities.add(ident)
        exchange=a.get('exchange')
        if a.get('status')!='active':
            reason='inactive'
        elif exchange not in EXCHANGES:
            reason='venue_calendar_not_supported'
        else:
            universe.setdefault(EXCHANGES[exchange],{})[symbol]=kind
            continue
        excluded.append({'ticker':symbol,'reason':reason})
    if not universe:
        raise ValueError('No supported instruments')
    return universe,excluded


def main():
    try:
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        universe,excluded=discover(asset_client(key,secret))
        result=scan(calendar_client(key,secret),client(key,secret),universe=universe)
        result.update(candidate_count=len(CANDIDATES),discovery_excluded=excluded,
                      scan_mode='expansion_qualification')
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        error={'state':'failed','error_type':type(exc).__name__}
        if isinstance(exc,HTTPError):
            error['http_status']=exc.code
        print(json.dumps(error))
        raise SystemExit(1)
    if result['state']!='ranked':
        raise SystemExit(2)

if __name__=='__main__':
    main()
