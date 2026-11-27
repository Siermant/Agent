"""Read-only first market ranking from qualified Alpaca paper data.
No allocation, orders, ledger, or credential persistence.
"""
import argparse
import csv
import getpass
import json
import sys
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from . import alpaca_data, equity_data, equity_rank, fx_reference
from .equity_preflight import calendar_client, parse_calendar, VENUES


def write(path, fields, data):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(fields)
        w.writerows(data)


def scan(calendar_request, bar_request, fx_request=None, clock=None, universe=None):
    universe = universe if universe is not None else {mic:{s:'etf' if s=='QQQ' else 'stock' for s in symbols} for mic,symbols in VENUES.items()}
    if not universe or not set(universe)<=set(VENUES) or not 1<=sum(len(x) for x in universe.values())<=30:
        raise ValueError('Invalid research universe')
    if len({s for items in universe.values() for s in items})!=sum(len(x) for x in universe.values()):
        raise ValueError('Duplicate ticker')
    clock=clock or (lambda:datetime.now(timezone.utc))
    now=clock()
    end=now.astimezone(alpaca_data.NY).date()-timedelta(days=1)
    start=end-timedelta(days=119)
    manifest, sessions=[],[]
    for mic,symbols in universe.items():
        calendar=parse_calendar(calendar_request(mic,start,end),mic,start,end,now)
        sessions.extend((mic,str(d),cl.isoformat()) for d,cl in sorted(calendar.items()))
        manifest.extend(('alpaca:'+mic+':'+s,s,mic,kind) for s,kind in sorted(symbols.items()))
    with tempfile.TemporaryDirectory(prefix='gta-equity-scan-') as t:
        root=Path(t)
        m,c,b,f=(root/x for x in ('manifest.csv','calendar.csv','bars.csv','fx.csv'))
        write(m,alpaca_data.MANIFEST,manifest)
        write(c,equity_data.CAL_FIELDS,sessions)
        data=alpaca_data.collect(m,c,bar_request,now)
        alpaca_data.write_csv(b,data)
        kwargs={} if fx_request is None else {'request':fx_request}
        fx_reference.export(start,end,f,today=now.date(),**kwargs)
        checked=max(clock(),datetime.now(timezone.utc))
        dataset=equity_data.load(c,b,f,now=checked)
        result=equity_rank.rank(dataset,now=checked)
    result['scan_mode']='manual_read_only'
    result['expected_instruments']=len(manifest)
    result['state']='ranked' if sum(len(x['instruments']) for x in result['cohorts'])==len(manifest) else 'incomplete_ranking'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt',action='store_true',required=True)
    p.parse_args()
    if not sys.stdin.isatty():
        p.error('Interactive terminal required')
    warnings.filterwarnings('error',category=getpass.GetPassWarning)
    try:
        key=getpass.getpass('Alpaca PAPER API key: ')
        secret=getpass.getpass('Alpaca PAPER secret: ')
        result=scan(calendar_client(key,secret),alpaca_data.client(key,secret))
    except Exception as exc:
        p.exit(1,'Market scan failed: '+type(exc).__name__+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    if result['state']!='ranked':
        raise SystemExit(2)


if __name__=='__main__':
    main()
