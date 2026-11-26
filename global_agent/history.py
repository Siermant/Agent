"""Download closed UTC daily OKX EUR spot bars and approximate PLN research CSV.
Uses strictly earlier NBP daily reference FX (not executable FX quotes).
No live orders, no API credentials.
"""
import argparse
import csv
import json
import os
import tempfile
from bisect import bisect_left
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from .ledger import positive
from .market import SYMBOLS

OKX = 'https://eea.okx.com/api/v5/market/history-candles'
NBP = 'https://api.nbp.pl/api/exchangerates/rates/a/eur/'
DAY = timedelta(days=1)


def get_json(url):
    request = Request(url, headers={'Accept':'application/json','User-Agent':'GlobalTradingAgent/0.5'})
    with urlopen(request, timeout=20) as response:
        data = response.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError('Oversized API response')
    return json.loads(data, parse_float=Decimal)


def candle_page(payload):
    if payload.get('code') != '0' or not isinstance(payload.get('data'), list):
        raise ValueError('OKX history request rejected')
    rows = {}
    stamps = []
    for row in payload['data']:
        if len(row) != 9:
            raise ValueError('Unexpected candle format')
        stamp = int(row[0])
        if stamp % 86400000:
            raise ValueError('Not a UTC daily candle')
        stamps.append(stamp)
        if row[8] == '0':
            continue
        if row[8] != '1':
            raise ValueError('Unknown candle confirmation flag')
        o,h,l,c = map(positive, row[1:5])
        if not l <= min(o,c) <= max(o,c) <= h:
            raise ValueError('Inconsistent OHLC')
        day = datetime.fromtimestamp(stamp/1000, timezone.utc).date()
        if day in rows and rows[day] != (o,c):
            raise ValueError('Conflicting duplicate candle')
        rows[day] = (o,c)
    return rows, stamps


def fetch_symbol(symbol, start, end, request=get_json):
    if symbol not in SYMBOLS:
        raise ValueError('Unsupported EUR spot pair')
    cursor = int(datetime.combine(end+DAY, time(), timezone.utc).timestamp()*1000)
    rows = {}
    for _ in range(20):
        page, stamps = candle_page(request(OKX+'?'+urlencode({'instId':symbol,'bar':'1Dutc','after':str(cursor),'limit':'100'})))
        if not stamps:
            break
        for day, prices in page.items():
            if start <= day <= end:
                if day in rows and rows[day] != prices:
                    raise ValueError('Conflicting candles across pages')
                rows[day] = prices
        oldest = min(stamps)
        if oldest >= cursor:
            raise ValueError('Pagination did not advance')
        cursor = oldest
        if datetime.fromtimestamp(cursor/1000,timezone.utc).date() <= start:
            break
    expected = {start+timedelta(days=i) for i in range((end-start).days+1)}
    if set(rows) != expected:
        raise ValueError(f'{symbol}: missing {len(expected-set(rows))} closed daily bars; no gap filling')
    return rows


def fetch_fx(start, end, request=get_json):
    cursor = start-timedelta(days=10)
    rates = {}
    while cursor <= end:
        last = min(end,cursor+timedelta(days=89))
        try:
            payload = request(f'{NBP}{cursor}/{last}/?format=json')
        except HTTPError as exc:
            if exc.code != 404:
                raise
            payload = {'code':'EUR','rates':[]}
        if payload.get('code') != 'EUR' or not isinstance(payload.get('rates'),list):
            raise ValueError('Invalid NBP response')
        for row in payload['rates']:
            day = date.fromisoformat(row['effectiveDate'])
            if not cursor <= day <= last:
                raise ValueError('NBP date outside requested interval')
            rates[day] = positive(row['mid'])
        cursor = last+DAY
    return rates


def prior_rate(day, rates, ordered):
    i = bisect_left(ordered,day)-1
    if i < 0 or day-ordered[i] > timedelta(days=7):
        raise ValueError(f'Missing recent prior NBP rate for {day}')
    return ordered[i], rates[ordered[i]]


def download(start, end, symbols, output, request=get_json):
    if end >= datetime.now(timezone.utc).date() or not 20 <= (end-start).days <= 729:
        raise ValueError('Use 21–730 completed UTC days')
    symbols = sorted(set(symbols))
    if not symbols or not set(symbols) <= SYMBOLS:
        raise ValueError('Unsupported or empty symbol list')
    rates = fetch_fx(start,end,request)
    ordered = sorted(rates)
    series = {s:fetch_symbol(s,start,end,request) for s in symbols}
    output = Path(output)
    # Write only after all data checks pass, then atomically replace output.
    output.parent.mkdir(parents=True,exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',dir=output.parent,delete=False) as f:
            name = f.name
            writer = csv.writer(f)
            writer.writerow(['date','symbol','open_pln','close_pln','open_eur','close_eur','fx_date','eur_pln','source'])
            for day in sorted(series[symbols[0]]):
                fx_day, rate = prior_rate(day,rates,ordered)
                for s in symbols:
                    o,c = series[s][day]
                    writer.writerow([day,s,o*rate,c*rate,o,c,fx_day,rate,'OKX_1Dutc_NBP_A_prior'])
        os.replace(name,output)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)
    return len(series[symbols[0]])*len(symbols)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',type=date.fromisoformat,required=True)
    p.add_argument('--end',type=date.fromisoformat,required=True)
    p.add_argument('--symbols',nargs='+',default=['BTC-EUR','ETH-EUR'])
    p.add_argument('--output',required=True)
    a = p.parse_args()
    print(f'Rows exported: {download(a.start,a.end,a.symbols,a.output)}')

if __name__ == '__main__':
    main()
