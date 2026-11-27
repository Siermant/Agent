"""Read-only NBP Table A FX export, independent of the active service."""
import argparse
import csv
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from .history import get_json
from .equity_data import positive, FX_FIELDS

BASE = 'https://api.nbp.pl/api/exchangerates/rates/a/'
DAY = timedelta(days=1)


def rates(currency, start, end, request=get_json):
    if currency not in ('USD','EUR') or not 0 <= (end-start).days <= 729:
        raise ValueError('Unsupported currency or date range')
    result = {}
    cursor = start
    while cursor <= end:
        last = min(end, cursor+timedelta(days=89))
        try:
            payload = request(f'{BASE}{currency.lower()}/{cursor}/{last}/?format=json')
        except HTTPError as exc:
            if exc.code != 404:
                raise
            payload = {'table':'A','code':currency,'rates':[]}
        if payload.get('table')!='A' or payload.get('code')!=currency or not isinstance(payload.get('rates'),list):
            raise ValueError('Unexpected NBP response')
        for row in payload['rates']:
            day = date.fromisoformat(row['effectiveDate'])
            if not cursor <= day <= last or day in result:
                raise ValueError('Unexpected or duplicate FX date')
            result[day] = positive(str(row['mid']))
        cursor = last+DAY
    return result


def export(start, end, output, request=get_json, today=None):
    today = today or datetime.now(timezone.utc).date()
    if end >= today or not 0 <= (end-start).days <= 719:
        raise ValueError('Use up to 720 completed calendar days')
    values = []
    for currency in ('EUR','USD'):
        data = rates(currency,start-timedelta(days=10),end,request)
        if not data or not any(0 < (start-d).days <= 7 for d in data):
            raise ValueError('Missing opening prior FX for '+currency)
        if not any(0 <= (end-d).days <= 7 for d in data):
            raise ValueError('Missing recent closing FX for '+currency)
        values.extend((currency,str(d),str(v)) for d,v in sorted(data.items()))
    output = Path(output)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',dir=output.parent,delete=False) as f:
            name=Path(f.name)
            writer=csv.writer(f)
            writer.writerow(FX_FIELDS)
            writer.writerows(values)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,output)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)
    return len(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',type=date.fromisoformat,required=True)
    p.add_argument('--end',type=date.fromisoformat,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print('NBP reference rates exported:',export(a.start,a.end,a.output))

if __name__=='__main__':
    main()
