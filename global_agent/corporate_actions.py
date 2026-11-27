"""Read-only corporate-action feed qualification, never an accounting operation.

The provider filters by process date, not ex-date. Coverage and timeliness
are not guaranteed. A successful response is NOT proof of no corporate action.
"""
import json
import os
import re
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from urllib.parse import urlencode
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError
from uuid import UUID
from .alpaca_data import NoRedirect, NY
from .equity_job import credentials
from .equity_quotes import SYMBOLS, universe
from .equity_data import positive

ENDPOINT='https://data.alpaca.markets/v1/corporate-actions'
SIMPLE={'cash_dividends','forward_splits','reverse_splits'}
KNOWN=SIMPLE|{'unit_splits','stock_dividends','spin_offs','spinoffs','cash_mergers',
    'stock_mergers','stock_and_cash_mergers','redemptions','name_changes',
    'worthless_removals','rights_distributions','partial_calls','reorganizations',
    'capital_gains_distributions'}
SYMBOL_FIELDS=('symbol','old_symbol','new_symbol','source_symbol','acquirer_symbol',
               'acquiree_symbol','alternate_symbol','target_symbol','parent_symbol','child_symbol')
DATE_FIELDS=('ex_date','effective_date','payable_date','record_date','process_date',
             'due_bill_on_date','due_bill_off_date','due_bill_redemption_date')


def client(key,secret):
    if not key or not secret or any(c.isspace() for c in key+secret):
        raise ValueError('Invalid data credentials')
    opener=build_opener(ProxyHandler({}),NoRedirect())
    def get(params):
        req=Request(ENDPOINT+'?'+urlencode(params),headers={
            'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret,
            'Accept':'application/json','User-Agent':'GTA-research'})
        with opener.open(req,timeout=20) as response:
            raw=response.read(2_000_001)
        if len(raw)>2_000_000:
            raise ValueError('Oversized corporate action response')
        return json.loads(raw,parse_float=D)
    return get


def normalize(kind,row,selected,start,end):
    if not isinstance(row,dict):
        raise ValueError('Invalid event')
    ident=str(UUID(row['id']))
    matched=sorted({row.get(k) for k in SYMBOL_FIELDS if isinstance(row.get(k),str) and row[k] in selected})
    if not matched:
        raise ValueError('Unmatched corporate action')
    dates={}
    for k in DATE_FIELDS:
        v=row.get(k)
        if v not in (None,''):
            d=date.fromisoformat(v)
            if d.isoformat()!=v:
                raise ValueError('Invalid date format')
            dates[k]=v
    if 'process_date' in dates and not start<=date.fromisoformat(dates['process_date'])<=end:
        raise ValueError('Unexpected process date')
    event={'id':ident,'type':kind,'symbols':matched,'dates':dates,
           'state':'manual_review_required'}
    # Incomplete data and all complex event types are retained as review items.
    # No currency guessing, no tax model, no implicit fractional-share handling.
    if kind not in SIMPLE or 'process_date' not in dates or 'ex_date' not in dates:
        return event
    if row.get('symbol') not in selected or len(matched)!=1:
        return event
    try:
        if kind=='cash_dividends':
            if (row.get('currency')!='USD' or row.get('foreign') is not False
                    or row.get('special') is not False or row.get('sub_type') not in (None,'')
                    or any(k.startswith('due_bill') for k in dates)
                    or 'payable_date' not in dates or dates['payable_date']<dates['ex_date']):
                return event
            amount=positive(str(row['rate']))
            if amount>D(10000):
                return event
            event.update(rate_usd=str(amount),state='simple_dividend_candidate')
        else:
            if any(k.startswith('due_bill') for k in dates):
                return event
            old,new=positive(str(row['old_rate'])),positive(str(row['new_rate']))
            ratio=new/old
            if not D('.0001')<=ratio<=D(10000) or (kind=='forward_splits' and ratio<=1) or (kind=='reverse_splits' and ratio>=1):
                return event
            event.update(old_rate=str(old),new_rate=str(new),state='simple_split_candidate')
    except (KeyError,ValueError,ArithmeticError,TypeError):
        pass
    return event


def scan(request,symbols,start,end):
    selected=universe(symbols)
    if not isinstance(start,date) or not isinstance(end,date) or not 0<=(end-start).days<=180:
        raise ValueError('Invalid process-date window')
    cursor=start
    found={}
    pages=0
    while cursor<=end:
        last=min(end,cursor+timedelta(days=59))
        params={'symbols':','.join(selected),'region':'us','start':str(cursor),'end':str(last),
                'data_quality':'all','limit':100,'sort':'asc'}
        tokens=set()
        for _ in range(50):
            payload=request(dict(params))
            pages+=1
            groups=payload['corporate_actions']
            if not isinstance(groups,dict):
                raise ValueError('Invalid corporate action groups')
            for kind,rows in groups.items():
                if kind not in KNOWN or not isinstance(rows,list):
                    raise ValueError('Unknown corporate action type')
                for row in rows:
                    e=normalize(kind,row,selected,cursor,last)
                    if e['id'] in found:
                        raise ValueError('Duplicate corporate action ID')
                    found[e['id']]=e
                    if len(found)>3000:
                        raise ValueError('Too many corporate actions')
            token=payload['next_page_token']
            if token is None:
                break
            if not isinstance(token,str) or not token or len(token)>4096 or token in tokens:
                raise ValueError('Invalid pagination token')
            tokens.add(token)
            params['page_token']=token
        else:
            raise ValueError('Pagination limit exceeded')
        cursor=last+timedelta(days=1)
    events=sorted(found.values(),key=lambda e:(e['dates'].get('ex_date',e['dates'].get('effective_date',e['dates'].get('process_date',''))),e['id']))
    counts={k:sum(e['state']==k for e in events) for k in ('simple_dividend_candidate','simple_split_candidate','manual_review_required')}
    return {'state':'feed_qualified','research_only':True,'real_trading':False,'eligible_for_orders':False,
            'corporate_actions_verified':False,'coverage_guaranteed':False,'accounting_enabled':False,
            'date_basis':'provider_process_date','process_start':str(start),'process_end':str(end),
            'symbols_checked':len(selected),'pages':pages,'events_count':len(events),'counts':counts,
            'events':events}


def main():
    try:
        key,secret=credentials(os.environ['CREDENTIALS_DIRECTORY'])
        today=datetime.now(timezone.utc).astimezone(NY).date()
        result=scan(client(key,secret),SYMBOLS,today-timedelta(days=60),today+timedelta(days=30))
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        error={'state':'failed','error_type':type(exc).__name__,'real_trading':False}
        if isinstance(exc,HTTPError):
            error['http_status']=exc.code
        print(json.dumps(error))
        raise SystemExit(1)

if __name__=='__main__':
    main()
