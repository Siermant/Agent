import unittest
from datetime import datetime, timezone, timedelta
from urllib.parse import urlsplit
from global_agent.equity_scan import scan
from global_agent.alpaca_data import NY
from global_agent.equity_preflight import VENUES

def calendar(mic,start,end):
    data=[]
    while start<=end:
        if start.weekday()<5:
            dt=datetime.combine(start,datetime.min.time(),NY)
            data.append({'date':str(start),'core_start':dt.replace(hour=9,minute=30).isoformat(),
                         'core_end':dt.replace(hour=16).isoformat()})
        start+=timedelta(days=1)
    return {'market':{'mic':mic},'calendar':data}

def bars(p):
    start=datetime.fromisoformat(p['start']).date()
    end=datetime.fromisoformat(p['end']).date()
    data=[]
    i=0
    while start<=end:
        if start.weekday()<5:
            v=str(100+i)
            data.append({'t':datetime.combine(start,datetime.min.time(),NY).isoformat(),'o':v,'h':v,'l':v,'c':v})
            i+=1
        start+=timedelta(days=1)
    return {'bars':{s:data for s in p['symbols'].split(',')},'next_page_token':None}

def fx(url):
    cur,a,b=urlsplit(url).path.strip('/').split('/')[-3:]
    start,end=datetime.fromisoformat(a).date(),datetime.fromisoformat(b).date()
    data=[]
    while start<=end:
        data.append({'effectiveDate':str(start),'mid':'4'})
        start+=timedelta(days=1)
    return {'table':'A','code':cur.upper(),'rates':data}

class EquityScanTests(unittest.TestCase):
    def test_scan(self):
        r=scan(calendar,bars,fx)
        self.assertEqual(r['state'],'ranked')
        self.assertEqual(r['expected_instruments'],4)
        self.assertEqual(len(r['cohorts']),2)
        self.assertFalse(r['eligible_for_orders'])
        self.assertEqual(sum(len(x['instruments']) for x in r['cohorts']),4)
    def test_missing_data(self):
        with self.assertRaises(ValueError):
            scan(calendar,lambda _: {'bars':{},'next_page_token':None},fx)
