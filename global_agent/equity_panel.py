"""Read-only equity report validation and escaped HTML fragment for the dashboard."""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from html import escape
import json
from pathlib import Path
from .equity_data import moment

MAX_BYTES=256*1024
MAX_AGE=timedelta(hours=30)


def read_json(path):
    with path.open(encoding='utf-8') as f:
        text=f.read(MAX_BYTES+1)
    if len(text)>MAX_BYTES:
        raise ValueError('Oversized report')
    return json.loads(text)


def summary(directory, now=None):
    if directory is None:
        return {'state':'not_connected','automated_trading':False}
    now=now or datetime.now(timezone.utc)
    root=Path(directory)
    base={'automated_trading':False,'research_only':True}
    try:
        status=read_json(root/'status.json')
        ranking=read_json(root/'ranking.json')
        for obj in (status,ranking):
            if (obj.get('research_only') is not True or obj.get('real_trading') is not False
                    or obj.get('eligible_for_orders') is not False):
                raise ValueError('Unsafe report flags')
        if status.get('state')!='ok' or ranking.get('state')!='ranked':
            return dict(base,state='scan_not_ok')
        generated=moment(ranking['generated_utc'])
        finished=moment(status['finished_utc'])
        if status['ranking_generated_utc']!=ranking['generated_utc'] or not generated<=finished<=now:
            raise ValueError('Inconsistent report timestamps')
        if not timedelta(0)<=now-generated<=MAX_AGE:
            return dict(base,state='stale',generated_utc=generated.isoformat())
        cohorts=[]
        seen=set()
        for c in ranking['cohorts']:
            mic=c['mic']
            if mic not in ('XNAS','XNYS') or c['source']!='alpaca_iex_daily':
                raise ValueError('Unsupported cohort')
            rows=[]
            for i,r in enumerate(c['instruments'],1):
                if r['instrument_id'] in seen or r['mic']!=mic or r['rank']!=i:
                    raise ValueError('Invalid instrument')
                seen.add(r['instrument_id'])
                values={}
                for k in ('return_20_pct','return_60_pct','annualized_volatility_pct','max_drawdown_60_pct','score'):
                    v=Decimal(str(r[k]))
                    if not v.is_finite() or abs(v)>1000000:
                        raise ValueError('Invalid metric')
                    values[k]=str(v.quantize(Decimal('.01')))
                rows.append(dict(values,ticker=r['ticker'],rank=i))
            cohorts.append({'mic':mic,'last_session':c['last_session'],'instruments':rows})
        if not 1<=len(seen)<=30 or len(seen)!=status['instruments'] or len(seen)!=ranking['expected_instruments'] or len(cohorts)!=status['cohorts']:
            raise ValueError('Inconsistent report counts')
        return dict(base,state='ok',generated_utc=generated.isoformat(),
                    instruments=len(seen),cohorts=cohorts)
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError,AttributeError):
        return dict(base,state='unavailable')


def render_equities(result):
    labels={'not_connected':'Oczekiwanie na połączenie z modułem giełdowym',
            'scan_not_ok':'Ostatnia analiza nie zakończyła się poprawnie',
            'stale':'Raport nieaktualny','unavailable':'Brak spójnego raportu',
            'ok':'Analiza badawcza — bez zleceń'}
    out=['<section><h2>Akcje i ETF — analiza badawcza</h2><p>',
         escape(labels.get(result.get('state'),'Stan nieznany')),
         '</p><p>Ranking historyczny, nie rekomendacja. IKE i handel akcjami nieaktywne.</p>']
    if result.get('state')=='ok':
        out+=['<p>Ostatnia analiza: ',escape(result['generated_utc']),'</p>']
        for cohort in result['cohorts']:
            out+=['<h3>',escape(cohort['mic']),' — ',escape(cohort['last_session']),
                  '</h3><div class="scroll"><table><thead><tr><th>Instrument</th><th>Ocena</th><th>20 sesji %</th><th>60 sesji %</th><th>Zmienność %</th><th>Obsunięcie %</th></tr></thead><tbody>']
            for r in cohort['instruments']:
                out.append('<tr>'+''.join('<td>'+escape(str(r[k]))+'</td>' for k in ('ticker','score','return_20_pct','return_60_pct','annualized_volatility_pct','max_drawdown_60_pct'))+'</tr>')
            out.append('</tbody></table></div>')
    out.append('</section>')
    return ''.join(out)
