"""Scheduled local equity research only. Reads systemd credentials, never orders."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from .equity_scan import scan
from .equity_preflight import calendar_client
from .alpaca_data import client
from .research_cycle import publish
from .equity_export import export


def credentials(directory):
    path=Path(directory)/'alpaca-paper.json'
    with path.open(encoding='utf-8') as f:
        raw=f.read(4097)
    if len(raw)>4096:
        raise ValueError('Oversized credential')
    obj=json.loads(raw)
    if set(obj)!= {'key','secret'}:
        raise ValueError('Invalid credential fields')
    for value in obj.values():
        if not isinstance(value,str) or not value or any(c.isspace() for c in value):
            raise ValueError('Invalid credential')
    return obj['key'],obj['secret']


def cycle(state_dir, fetch, clock=None, export_dir=None):
    clock=clock or (lambda:datetime.now(timezone.utc))
    root=Path(state_dir)
    started=clock().isoformat()
    status={'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'started_utc':started,'state':'running'}
    publish(status,root/'status.json')
    try:
        if export_dir is not None:
            export(export_dir,status)
        result=fetch()
        if (result.get('research_only') is not True or result.get('real_trading') is not False
                or result.get('eligible_for_orders') is not False or result.get('state')!='ranked'):
            raise ValueError('Incomplete or invalid ranking')
        count=sum(len(c['instruments']) for c in result['cohorts'])
        if count!=result['expected_instruments'] or count<1:
            raise ValueError('Incomplete instrument set')
        result=dict(result,scan_mode='scheduled_read_only')
        publish(result,root/'ranking.json')
        status.update(state='ok',instruments=count,cohorts=len(result['cohorts']),
                      ranking_generated_utc=result['generated_utc'])
    except Exception as exc:
        status.update(state='failed',error_type=type(exc).__name__)
        if isinstance(exc,HTTPError):
            status['http_status']=exc.code
    status['finished_utc']=clock().isoformat()
    if export_dir is not None:
        try:
            export(export_dir,status,result if status['state']=='ok' else None)
        except Exception as exc:
            status['panel_export_error']=type(exc).__name__
            status['state']='failed'
    publish(status,root/'status.json')
    return status


# Listing membership confirmed by the operator's 2026-11-27 qualification.
QUALIFIED = {
    'XNAS': frozenset('WMT COST AAPL TLT PEP HON AMZN MSFT META QQQ GOOGL AVGO AMD NVDA'.split()),
    'XNYS': frozenset('KO MA V ABBV JNJ IBM CVX GE XOM JPM BAC UNH CAT ORCL'.split()),
}


def expanded_scan(asset_request, calendar_request, bar_request, scanner=scan):
    # Local import avoids a circular import of the credentials helper.
    from .equity_expand import discover, CANDIDATES
    universe, excluded = discover(asset_request)
    expected = {mic:{symbol:CANDIDATES[symbol] for symbol in symbols}
                for mic,symbols in QUALIFIED.items()}
    if universe != expected or {x['ticker']:x['reason'] for x in excluded} != {
            'SPY':'venue_calendar_not_supported', 'VT':'venue_calendar_not_supported'}:
        raise ValueError('Qualified universe changed')
    result = scanner(calendar_request,bar_request,universe=universe)
    observed = {}
    seen = set()
    for cohort in result['cohorts']:
        for row in cohort['instruments']:
            symbol, mic = row['ticker'], row['mic']
            if symbol in seen or mic != cohort['mic']:
                raise ValueError('Duplicate or mismatched ranked instrument')
            seen.add(symbol)
            observed.setdefault(mic,{})[symbol]=row['kind']
    if observed != expected or result['expected_instruments'] != len(seen):
        raise ValueError('Incomplete qualified ranking')
    return dict(result,candidate_count=len(CANDIDATES),discovery_excluded=excluded)


def main():
    try:
        root=Path(os.environ['STATE_DIRECTORY'])
        def fetch():
            key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
            from .equity_expand import asset_client
            return expanded_scan(asset_client(key,secret),calendar_client(key,secret),client(key,secret))
        status=cycle(root,fetch,export_dir=os.getenv('CACHE_DIRECTORY'))
        print(json.dumps(status,allow_nan=False))
    except Exception as exc:
        print(json.dumps({'state':'failed','error_type':type(exc).__name__}))
        raise SystemExit(1)
    if status['state']!='ok':
        raise SystemExit(1)


if __name__=='__main__':
    main()
