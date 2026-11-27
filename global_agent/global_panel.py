"""Bounded local Global paper dashboard report, never a ledger or order interface."""
import json
import os
import tempfile
import re
from datetime import datetime,timedelta,timezone
from decimal import Decimal as D
from pathlib import Path
from html import escape
from .equity_data import moment
from .equity_panel import read_json

ACCOUNTS=tuple(p+':global' for p in 'ABCDE')
FLAGS={'research_only':True,'real_trading':False,'eligible_for_orders':False,
       'simulated':True,'rebalance_enabled':False,'corporate_actions_verified':False,
       'return_type':'price_only_reference_fx'}
STATES={'valued','market_closed','waiting_for_initial_open','incomplete_quotes',
        'session_changed','quotes_unavailable','failed'}
MONEY=('capital_pln','cash_pln','invested_pln','equity_pln','pnl_pln','return_pct','drawdown_pct')
FIELDS=('account',)+MONEY+('risk_blocked','fills','positions_count')
LIMIT=256*1024


def select(obj,keys):
    return {k:obj[k] for k in keys}


def export_panel(directory,status,valuation=None):
    root=Path(directory)
    out={'status':select(status,tuple(FLAGS)+('state','checked_utc'))}
    if valuation is not None:
        out['valuation']=dict(select(valuation,tuple(FLAGS)+('state','checked_utc','valued_at','fx_date','quote_source')),
            **{k:valuation[k] for k in ('quote_delay_seconds','quote_observed_from_utc','quote_observed_to_utc') if k in valuation},
            accounts=[dict(select(r,FIELDS),positions=[select(p,('ticker','quantity','value_pln')) for p in r['positions']]) for r in valuation['accounts']],
            history=[select(r,('at','equity_pln')) for r in valuation['history']],
            fills=[select(r,('account','ticker','quantity','price_pln','fee_pln','at','side')) for r in valuation['fills']])
    data=json.dumps(out,allow_nan=False,ensure_ascii=True).encode()
    if len(data)>LIMIT:
        raise ValueError('Oversized global report')
    temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=root,delete=False) as f:
            temp=Path(f.name)
            f.write(data)
            f.flush()
            os.fchmod(f.fileno(),0o644)
            os.fsync(f.fileno())
        os.replace(temp,root/'global.json')
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def num(v,lo=D('-1e9'),hi=D('1e9')):
    n=D(str(v))
    if not n.is_finite() or not lo<=n<=hi:
        raise ValueError('Invalid metric')
    return n

def fmt(v):
    return str(num(v).quantize(D('.01')))

def ticker(v):
    if not isinstance(v,str) or not re.fullmatch(r'[A-Z][A-Z0-9.]{0,9}',v):
        raise ValueError('Invalid ticker')
    return v

def count(v,maximum):
    if type(v) is not int or not 0<=v<=maximum:
        raise ValueError('Invalid count')
    return v

def flags(obj):
    if any(type(obj.get(k)) is not type(v) or obj[k]!=v for k,v in FLAGS.items()):
        raise ValueError('Unsafe Global report flags')


def summary(directory,now=None):
    base=dict(FLAGS)
    if directory is None:
        return dict(base,state='not_connected')
    now=now or datetime.now(timezone.utc)
    try:
        report=read_json(Path(directory)/'global.json')
        status=report['status']
        flags(status)
        checked=moment(status['checked_utc'])
        state=status['state']
        if state not in STATES or checked>now:
            raise ValueError('Invalid report status')
        out=dict(base,state=state,checked_utc=checked.isoformat())
        if 'valuation' not in report:
            if state=='valued':
                raise ValueError('Missing valuation')
            return out
        v=report['valuation']
        flags(v)
        at=moment(v['valued_at'])
        source=v['quote_source']
        delays={'alpaca_iex_latest':0,'alpaca_delayed_sip_latest':900}
        if source not in delays or v.get('quote_delay_seconds',0)!=delays[source]:
            raise ValueError('Invalid quote source or delay')
        delay=delays[source]
        observed={}
        if delay or 'quote_observed_from_utc' in v or 'quote_observed_to_utc' in v:
            lo=moment(v['quote_observed_from_utc']);hi=moment(v['quote_observed_to_utc'])
            if not at-timedelta(seconds=delay+60)<=lo<=hi<=at-timedelta(seconds=delay):
                raise ValueError('Invalid observed quote range')
            observed={'quote_observed_from_utc':lo.isoformat(),'quote_observed_to_utc':hi.isoformat()}
        if v['state']!='valued' or not at<=checked or moment(v['checked_utc'])!=at:
            raise ValueError('Invalid valuation chronology')
        if state=='valued' and at!=checked:
            raise ValueError('Valuation/status mismatch')
        if now-at>timedelta(days=7):
            return dict(out,state='stale',valued_at=at.isoformat())
        accounts=[]
        for r in v['accounts']:
            if r['account'] not in ACCOUNTS or type(r['risk_blocked']) is not bool:
                raise ValueError('Invalid account')
            n={k:num(r[k]) for k in MONEY}
            if (n['capital_pln']!=50000 or min(n['cash_pln'],n['invested_pln'],n['equity_pln'])<0
                or abs(n['cash_pln']+n['invested_pln']-n['equity_pln'])>D('.01')
                or abs(n['equity_pln']-50000-n['pnl_pln'])>D('.01')
                or abs(n['pnl_pln']/500-n['return_pct'])>D('.001')
                or not 0<=n['drawdown_pct']<=100):
                raise ValueError('Inconsistent account metrics')
            positions=[{'ticker':ticker(p['ticker']),'quantity':str(num(p['quantity'],D(1))),
                        'value_pln':fmt(num(p['value_pln'],D(0)))} for p in r['positions']]
            if (len(positions)!=count(r['positions_count'],26) or len({p['ticker'] for p in positions})!=len(positions)
                or abs(sum((num(p['value_pln']) for p in positions),D(0))-n['invested_pln'])>D('.14')):
                raise ValueError('Inconsistent positions')
            accounts.append(dict({k:fmt(x) for k,x in n.items()},account=r['account'],risk_blocked=r['risk_blocked'],
                                 fills=count(r['fills'],26),positions=positions))
        if len(accounts)!=5 or {r['account'] for r in accounts}!=set(ACCOUNTS):
            raise ValueError('Missing accounts')
        history=[]
        last=None
        for h in v['history']:
            t=moment(h['at'])
            if t>at or (last is not None and t<=last) or set(h['equity_pln'])!=set(ACCOUNTS):
                raise ValueError('Invalid history')
            history.append({'at':t.isoformat(),'equity_pln':{k:fmt(num(x,D(0))) for k,x in h['equity_pln'].items()}})
            last=t
        if len(history)>31:
            raise ValueError('Excess history')
        fills=[]
        for f in v['fills']:
            if f['account'] not in ACCOUNTS or f['side']!='buy' or moment(f['at'])>at:
                raise ValueError('Invalid initial fill')
            fills.append({'account':f['account'],'ticker':ticker(f['ticker']),'side':'buy',
                          'quantity':str(num(f['quantity'],D(1))),'price_pln':fmt(num(f['price_pln'],D('.000001'))),
                          'fee_pln':fmt(num(f['fee_pln'],D(0))),'at':moment(f['at']).isoformat()})
        if len(fills)!=sum(a['fills'] for a in accounts) or len(fills)>104:
            raise ValueError('Invalid fill count')
        fx=datetime.strptime(v['fx_date'],'%Y-%m-%d').date()
        if not timedelta(0)<at.date()-fx<=timedelta(days=8):
            raise ValueError('Invalid FX date')
        return dict(out,valued_at=at.isoformat(),fx_date=fx.isoformat(),quote_source=source,quote_delay_seconds=delay,**observed,
                    fresh=state=='valued' and now-at<=timedelta(minutes=10),
                    accounts=sorted(accounts,key=lambda r:r['account']),history=history,fills=fills)
    except (OSError,ValueError,KeyError,TypeError,ArithmeticError,AttributeError):
        return dict(base,state='unavailable')


def chart(history):
    if not history:
        return '<p>Brak historii wycen.</p>'
    values=[[float(D(h['equity_pln'][a])/500-100) for h in history] for a in ACCOUNTS]
    lo=min(0,min(min(v) for v in values))
    hi=max(0,max(max(v) for v in values))
    if hi-lo<.1:
        hi=lo+.1
    colors=('#2563eb','#16a34a','#ea580c','#9333ea','#64748b')
    w,h=700,210
    def x(i):return 60+i*610/max(1,len(history)-1)
    def y(v):return 18+(hi-v)*165/(hi-lo)
    out=['<svg viewBox="0 0 700 210" role="img" aria-label="Zwrot portfeli Global w procentach">']
    for i in range(5):
        v=lo+(hi-lo)*i/4
        yy=y(v)
        out.append(f'<line x1="60" x2="670" y1="{yy:.1f}" y2="{yy:.1f}" stroke="#888" opacity=".3"/><text x="3" y="{yy+4:.1f}" font-size="11" fill="currentColor">{v:.2f}%</text>')
    for color,vs in zip(colors,values):
        points=' '.join(f'{x(i):.1f},{y(v):.1f}' for i,v in enumerate(vs))
        out.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        out.append(f'<circle cx="{x(len(vs)-1):.1f}" cy="{y(vs[-1]):.1f}" r="3" fill="{color}"/>')
    out.append('</svg><p>')
    out.extend(f'<span style="color:{c};font-weight:bold"> {a[0]} </span>' for a,c in zip(ACCOUNTS,colors))
    out.append('</p><p>Ostatnia wycena każdego dnia UTC; nie kurs zamknięcia.</p>')
    return ''.join(out)


def render_global(s):
    labels={'not_connected':'Moduł jeszcze niepodłączony','unavailable':'Brak spójnego raportu',
            'waiting_for_initial_open':'Oczekiwanie na pierwsze otwarcie',
            'market_closed':'Giełda zamknięta','valued':'Ostatnia wycena',
            'stale':'Raport nieaktualny','failed':'Błąd wyceny','incomplete_quotes':'Brak aktualnych cen',
            'session_changed':'Zmiana sesji','quotes_unavailable':'Brak cen'}
    out=['<section><h2>Global A–E — oddzielna symulacja</h2><p>',escape(labels.get(s.get('state'),'Stan nieznany')),
         '</p><p>Wyłącznie zmiana cen i referencyjny USD/PLN; brak dywidend i splitów. E pozostaje w gotówce.</p>']
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'
    if s.get('accounts'):
        if s.get('quote_delay_seconds')==900:
            out+=['<p><strong>Delayed SIP — dane opóźnione o 15 minut. Symulacja po cenach historycznych, nie wykonanie rynkowe.</strong></p>',
                  '<p>Czas notowań UTC: ',escape(s['quote_observed_from_utc']),' — ',escape(s['quote_observed_to_utc']),'</p>']
        out+=['<p>',escape(s['valued_at']), ' · ', 'Raport aktualny' if s['fresh'] else 'Ostatnia / nieaktualna',
              ' · NBP ',escape(s['fx_date']),'</p>',
              table(['Portfel','Gotówka PLN','Wartość PLN','Zysk/strata PLN','Zwrot %','Obsunięcie %','Ryzyko'],
                    [[r['account'],r['cash_pln'],r['equity_pln'],r['pnl_pln'],r['return_pct'],r['drawdown_pct'],'Blokada' if r['risk_blocked'] else 'OK'] for r in s['accounts']]),
              '<h3>Historia zwrotu</h3>',chart(s['history']),'<h3>Posiadane akcje</h3>',
              table(['Portfel','Ticker','Liczba','Wartość PLN'],[[a['account'],p['ticker'],p['quantity'],p['value_pln']] for a in s['accounts'] for p in a['positions']]),
              '<h3>Transakcje początkowe</h3>',
              table(['Czas UTC','Portfel','Ticker','Liczba','Cena PLN','Prowizja PLN'],[[f[k] for k in ('at','account','ticker','quantity','price_pln','fee_pln')] for f in s['fills']])]
    out.append('</section>')
    return ''.join(out)
