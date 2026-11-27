"""Alpaca IEX historical US equity data adapter. Research only, no orders."""
import argparse
import csv
import json
import os
import re
import tempfile
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
from zoneinfo import ZoneInfo
from .equity_data import FIELDS, CAL_FIELDS, rows, moment, positive

NY = ZoneInfo('America/New_York')
ENDPOINT = 'https://data.alpaca.markets/v2/stocks/bars'
MANIFEST = ('instrument_id','ticker','mic','kind')


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def client(key, secret):
    if not key or not secret or any(c.isspace() for c in key+secret):
        raise ValueError('Missing or invalid data credentials')
    opener = build_opener(ProxyHandler({}), NoRedirect())
    def request(params):
        req = Request(ENDPOINT+'?'+urlencode(params), headers={
            'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
            'Accept':'application/json','User-Agent':'GTA-research'})
        with opener.open(req, timeout=20) as response:
            raw=response.read(4_000_001)
        if len(raw)>4_000_000:
            raise ValueError('Oversized market response')
        return json.loads(raw,parse_float=Decimal)
    return request


def history(request, expected, adjustment, *, with_volume=False):
    if with_volume and adjustment!='raw':
        raise ValueError('Volume research requires raw bars')
    if adjustment not in ('raw','split,dividend'):
        raise ValueError('Unsupported adjustment')
    days = [d for _,d in expected]
    if not days or (max(days)-min(days)).days>365:
        raise ValueError('Use a bounded completed history window')
    symbols=sorted({s for s,_ in expected})
    if len(symbols)>30:
        raise ValueError('Too many symbols')
    params={'symbols':','.join(symbols),'timeframe':'1Day','feed':'iex',
            'currency':'USD','adjustment':adjustment,'limit':1000,'sort':'asc',
            'start':datetime.combine(min(days),time(),NY).isoformat(),
            'end':(datetime.combine(max(days)+timedelta(days=1),time(),NY)-timedelta(microseconds=1)).isoformat(),
            'asof':str(max(days))}
    found, tokens = {}, set()
    for _ in range(100):
        payload=request(dict(params))
        if not isinstance(payload.get('bars'),dict) or 'next_page_token' not in payload:
            raise ValueError('Invalid market response')
        for symbol,items in payload['bars'].items():
            if symbol not in symbols or not isinstance(items,list):
                raise ValueError('Unexpected symbol or bars')
            for item in items:
                stamp=moment(item['t']).astimezone(NY)
                key=symbol,stamp.date()
                if stamp.time()!=time() or key not in expected or key in found:
                    raise ValueError('Unexpected or duplicate session bar')
                o,h,l,c=(positive(str(item[k])) for k in ('o','h','l','c'))
                if not l<=min(o,c)<=max(o,c)<=h:
                    raise ValueError('Invalid OHLC')
                if with_volume:
                    v=item['v']
                    if type(v) is not int or not 0<v<=10**12:
                        raise ValueError('Invalid bar volume')
                    found[key]=(c,v,positive(str(item['vw'])))
                else:
                    found[key]=c
        token=payload['next_page_token']
        if token is None:
            if set(found)!=expected:
                raise ValueError('Missing requested symbol or session')
            return found
        if not isinstance(token,str) or not token or len(token)>4096 or token in tokens:
            raise ValueError('Invalid pagination token')
        tokens.add(token)
        params['page_token']=token
    raise ValueError('Pagination limit exceeded')


def collect(manifest, calendar, request, now=None):
    now=now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError('Now requires timezone')
    today=now.astimezone(NY).date()
    instruments={}
    ids=set()
    for row in rows(manifest, MANIFEST):
        ident,ticker,mic,kind=(row[k] for k in MANIFEST)
        if (not re.fullmatch('[A-Za-z0-9_.:-]{1,100}',ident) or
                not re.fullmatch('[A-Z][A-Z0-9.]{0,9}',ticker) or
                mic not in ('XNYS','XNAS','ARCX','BATS') or kind not in ('stock','etf') or
                ticker in instruments or ident in ids):
            raise ValueError('Invalid or duplicate US instrument')
        instruments[ticker]=row
        ids.add(ident)
    if not 1<=len(instruments)<=30:
        raise ValueError('Use 1–30 instruments')
    venues={r['mic'] for r in instruments.values()}
    sessions={}
    for row in rows(calendar,CAL_FIELDS):
        day=date.fromisoformat(row['session'])
        close=moment(row['close_at'])
        key=row['mic'],day
        if key[0] not in venues or key in sessions or day>=today or close>now or close.astimezone(NY).date()!=day:
            raise ValueError('Invalid supplied completed US calendar')
        sessions[key]=close
    if {m for m,_ in sessions}!=venues:
        raise ValueError('Missing venue calendar')
    expected={(s,d) for s,r in instruments.items() for m,d in sessions if r['mic']==m}
    raw=history(request,expected,'raw')
    adjusted=history(request,expected,'split,dividend')
    retrieved=max(now,datetime.now(timezone.utc))
    return [dict(instruments[s],currency='USD',exchange_tz=NY.key,session=str(d),
                 close_at=sessions[instruments[s]['mic'],d].isoformat(),retrieved_at=retrieved.isoformat(),
                 raw_close=str(raw[s,d]),adjusted_close=str(adjusted[s,d]),
                 adjustment='split_dividend',source='alpaca_iex_daily')
            for s,d in sorted(expected)]


def write_csv(output, data):
    output=Path(output)
    name=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',dir=output.parent,delete=False) as f:
            name=Path(f.name)
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader()
            w.writerows(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,output)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','calendar','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.resolve() in {a.manifest.resolve(),a.calendar.resolve()}:
        p.error('Output must differ from inputs')
    try:
        data=collect(a.manifest,a.calendar,client(os.getenv('APCA_API_KEY_ID'),os.getenv('APCA_API_SECRET_KEY')))
        write_csv(a.output,data)
    except Exception as exc:
        # Never echo provider response, request headers or credentials.
        p.exit(1,'Market import failed: '+type(exc).__name__+'\n')
    print(json.dumps({'research_only':True,'source':'alpaca_iex_daily','bars':len(data),'calendar_verified_externally':False}))


if __name__=='__main__':
    main()
