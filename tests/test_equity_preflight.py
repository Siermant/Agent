import unittest
from datetime import datetime,date,timedelta,timezone
from urllib.parse import urlsplit
from urllib.error import HTTPError
from global_agent.equity_preflight import qualify,parse_calendar,VENUES
from global_agent.alpaca_data import NY

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

def calendar(mic,start,end):
    items=[]
    d=start
    while d<=end:
        if d.weekday()<5:
            op=datetime.combine(d,datetime.min.time(),NY).replace(hour=9,minute=30)
            cl=op.replace(hour=16,minute=0)
            items.append({'date':str(d),'core_start':op.isoformat(),'core_end':cl.isoformat()})
        d+=timedelta(days=1)
    return {'market':{'mic':mic},'calendar':items}

def bars(p):
    start=datetime.fromisoformat(p['start']).date()
    end=datetime.fromisoformat(p['end']).date()
    rows=[]
    d=start
    while d<=end:
        if d.weekday()<5:
            stamp=datetime.combine(d,datetime.min.time(),NY).isoformat()
            rows.append({'t':stamp,'o':'100','h':'101','l':'99','c':'100'})
        d+=timedelta(days=1)
    return {'bars':{s:rows for s in p['symbols'].split(',')},'next_page_token':None}

def fx(url):
    cur,start,end=urlsplit(url).path.strip('/').split('/')[-3:]
    a,b=date.fromisoformat(start),date.fromisoformat(end)
    days=[]
    while a<=b:
        days.append({'effectiveDate':str(a),'mid':'4.2'})
        a+=timedelta(days=1)
    return {'table':'A','code':cur.upper(),'rates':days}

class EquityPreflightTests(unittest.TestCase):
    def test_success(self):
        r=qualify(calendar,bars,fx,NOW)
        self.assertTrue(r['ready'])
        self.assertFalse(r['eligible_for_orders'])
        self.assertFalse(r['independent_exchange_calendar_verified'])
        self.assertGreaterEqual(r['checks']['XNAS']['sessions'],61)
    def test_provider_failure_redacted(self):
        def denied(_):
            raise HTTPError('https://example.invalid',403,'SECRET',None,None)
        r=qualify(calendar,denied,fx,NOW)
        self.assertFalse(r['ready'])
        self.assertTrue(r['checks']['nbp']['ok'])
        self.assertEqual(r['checks']['XNAS']['http_status'],403)
        self.assertNotIn('SECRET',str(r))
    def test_wrong_venue(self):
        start=date(2026,8,1);end=date(2026,11,26)
        with self.assertRaises(ValueError):
            parse_calendar(calendar('XNYS',start,end),'XNAS',start,end,NOW)
    def test_early_close(self):
        start=date(2026,8,1);end=date(2026,11,26)
        p=calendar('XNAS',start,end)
        d=p['calendar'][-1]['date']
        p['calendar'][-1]['core_end']=d+'T13:00:00-05:00'
        result=parse_calendar(p,'XNAS',start,end,NOW)
        self.assertEqual(result[date.fromisoformat(d)].hour,18)
