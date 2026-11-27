"""Offline equity/ETF research import. No market orders or network calls."""
import argparse
import csv
import json
import re
import os
import tempfile
from bisect import bisect_left
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

FIELDS = ('instrument_id','ticker','mic','kind','currency','exchange_tz',
          'session','close_at','retrieved_at','raw_close','adjusted_close',
          'adjustment','source')
CAL_FIELDS = ('mic','session','close_at')
FX_FIELDS = ('currency','date','pln_per_unit')


def rows(path, fields):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError('Unexpected CSV columns')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed CSV row')
            yield row


def moment(value):
    t = datetime.fromisoformat(value)
    if t.utcoffset() is None:
        raise ValueError('Timestamp requires timezone')
    return t.astimezone(timezone.utc)


def positive(value):
    try:
        n = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError('Invalid price') from exc
    if not n.is_finite() or n <= 0:
        raise ValueError('Invalid price')
    return n


def load(calendar, bars, fx, *, now=None):
    now = now or datetime.now(timezone.utc)
    if now.utcoffset() is None:
        raise ValueError('Now requires timezone')
    sessions = {}
    for r in rows(calendar, CAL_FIELDS):
        key = r['mic'], date.fromisoformat(r['session'])
        close = moment(r['close_at'])
        if not re.fullmatch('[A-Z0-9]{4}',key[0]) or key in sessions or close > now:
            raise ValueError('Invalid or duplicate completed session')
        sessions[key] = close
    if not sessions:
        raise ValueError('Empty calendar')
    rates = {}
    for r in rows(fx, FX_FIELDS):
        key = r['currency'], date.fromisoformat(r['date'])
        if key[0] not in ('EUR','USD') or key in rates or key[1] > now.date():
            raise ValueError('Invalid or duplicate FX rate')
        rates[key] = positive(r['pln_per_unit'])
    dates = {c:sorted(d for cur,d in rates if cur==c) for c in ('EUR','USD')}
    meta, seen, result = {}, set(), []
    for r in rows(bars, FIELDS):
        ident, mic, cur = r['instrument_id'], r['mic'], r['currency']
        day = date.fromisoformat(r['session'])
        key = ident, day
        if (not re.fullmatch('[A-Za-z0-9_.:-]{1,100}',ident) or not r['ticker'] or
                not r['source'] or r['kind'] not in ('stock','etf') or cur not in ('EUR','USD','PLN') or key in seen):
            raise ValueError('Invalid instrument or duplicate bar')
        close, retrieved = moment(r['close_at']), moment(r['retrieved_at'])
        if sessions.get((mic,day)) != close or close.astimezone(ZoneInfo(r['exchange_tz'])).date()!=day or not close <= retrieved <= now:
            raise ValueError('Bar outside completed session or invalid retrieval time')
        signature = tuple(r[k] for k in ('ticker','mic','kind','currency','exchange_tz','adjustment','source'))
        if meta.setdefault(ident,signature)!=signature:
            raise ValueError('Inconsistent instrument metadata')
        raw = positive(r['raw_close'])
        adjusted = None
        if r['adjustment']=='none' and not r['adjusted_close']:
            pass
        elif r['adjustment'] in ('split','split_dividend') and r['adjusted_close']:
            adjusted = positive(r['adjusted_close'])
        else:
            raise ValueError('Inconsistent adjustment policy')
        if cur=='PLN':
            fx_day, rate = None, Decimal(1)
        else:
            i = bisect_left(dates[cur],day)-1
            if i<0 or day-dates[cur][i]>timedelta(days=7):
                raise ValueError('Missing recent prior reference FX')
            fx_day = dates[cur][i]
            rate = rates[cur,fx_day]
        seen.add(key)
        result.append(dict(r, raw_close_pln=str(raw*rate),
                           adjusted_close_pln=None if adjusted is None else str(adjusted*rate),
                           fx_date=None if fx_day is None else str(fx_day),
                           pln_per_unit=str(rate)))
    if not result:
        raise ValueError('Empty bar data')
    for ident, signature in meta.items():
        expected = {(ident,d) for venue,d in sessions if venue==signature[1]}
        actual = {key for key in seen if key[0]==ident}
        if expected!=actual:
            raise ValueError('Missing supplied calendar sessions for '+ident)
    return {'research_only':True,'real_trading':False,'calendar_verified_externally':False,
            'instruments':len(meta),'bars':sorted(result,key=lambda r:(r['instrument_id'],r['session']))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('calendar','bars','fx','output'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    if a.output.resolve() in {x.resolve() for x in (a.calendar,a.bars,a.fx)}:
        p.error('Output must differ from inputs')
    result=load(a.calendar,a.bars,a.fx)
    with tempfile.NamedTemporaryFile('w',dir=a.output.parent,delete=False,encoding='utf-8') as f:
        name=Path(f.name)
        try:
            json.dump(result,f,indent=2,allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
            os.replace(name,a.output)
        finally:
            name.unlink(missing_ok=True)
    print(json.dumps({'research_only':True,'instruments':result['instruments'],'bars':len(result['bars'])}))

if __name__=='__main__':
    main()
