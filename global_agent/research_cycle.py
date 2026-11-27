"""Run a complete read-only Global research cycle; never writes to the ledger."""
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone, date
from pathlib import Path

from . import equity_data, equity_rank, global_plan, alpaca_data, fx_reference


def research(calendar, bars, fx, assumptions, cohort=None, now=None):
    now = now or datetime.now(timezone.utc)
    dataset = equity_data.load(calendar, bars, fx, now=now)
    ranking = equity_rank.rank(dataset, now=now)
    if not ranking['cohorts']:
        return {'research_only': True, 'real_trading': False,
                'eligible_for_orders': False, 'generated_utc': now.isoformat(),
                'state': 'no_rankable_instruments', 'ranking': ranking, 'allocation': None}
    if cohort is None and len(ranking['cohorts']) != 1:
        return {'research_only': True, 'real_trading': False,
                'eligible_for_orders': False, 'generated_utc': now.isoformat(),
                'state': 'cohort_selection_required', 'ranking': ranking, 'allocation': None}
    allocation = global_plan.plan(ranking, assumptions, 0 if cohort is None else cohort, now=now)
    return {'research_only': True, 'real_trading': False,
            'eligible_for_orders': False, 'generated_utc': now.isoformat(),
            'state': 'hypothetical_plan', 'ranking': ranking, 'allocation': allocation}


def provider_cycle(manifest, calendar, assumptions, request, cohort=None,
                   fx_request=None):
    # No writes to live data, no raw provider data published.
    data = alpaca_data.collect(manifest, calendar, request)
    dates = [date.fromisoformat(r['session']) for r in data]
    with tempfile.TemporaryDirectory(prefix='gta-equity-') as tmp:
        root = Path(tmp)
        bars, fx = root/'bars.csv', root/'fx.csv'
        alpaca_data.write_csv(bars, data)
        kwargs = {} if fx_request is None else {'request':fx_request}
        fx_reference.export(min(dates), max(dates), fx, **kwargs)
        return research(calendar,bars,fx,assumptions,cohort)


def publish(result, output):
    output = Path(output)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, delete=False) as f:
            name = Path(f.name)
            json.dump(result,f,indent=2,allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,output)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calendar',required=True,type=Path)
    p.add_argument('--assumptions',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cohort',type=int)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bars',type=Path)
    mode.add_argument('--alpaca-manifest',type=Path)
    p.add_argument('--fx',type=Path)
    a = p.parse_args()
    if (a.bars is None) == (a.fx is not None):
        p.error('Offline mode requires bars and FX; provider mode downloads FX')
    sources = (a.calendar,a.assumptions,a.bars,a.fx,a.alpaca_manifest)
    if a.output.resolve() in {x.resolve() for x in sources if x is not None}:
        p.error('Output must differ from inputs')
    try:
        assumptions = json.loads(a.assumptions.read_text(encoding='utf-8'))
        if a.bars:
            result = research(a.calendar,a.bars,a.fx,assumptions,a.cohort)
        else:
            request = alpaca_data.client(os.getenv('APCA_API_KEY_ID'),os.getenv('APCA_API_SECRET_KEY'))
            result = provider_cycle(a.alpaca_manifest,a.calendar,assumptions,request,a.cohort)
        publish(result,a.output)
    except Exception as exc:
        p.exit(1,'Research cycle failed: '+type(exc).__name__+'\n')
    print(json.dumps({'state':result['state'],'real_trading':False,
                      'cohorts':len(result['ranking']['cohorts']),
                      'published':str(a.output)}))
    if result['state']!='hypothetical_plan':
        raise SystemExit(2)


if __name__=='__main__':
    main()
