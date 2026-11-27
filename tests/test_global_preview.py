import unittest
from datetime import date,timedelta
from decimal import Decimal as D
from uuid import uuid5,NAMESPACE_DNS
from global_agent.global_preview import preview,turnover
from global_agent.equity_job import QUALIFIED
from test_equity_scan import calendar,bars,fx

def asset(s):
    exchange='NASDAQ' if s in QUALIFIED['XNAS'] else 'NYSE' if s in QUALIFIED['XNYS'] else 'ARCA'
    return {'id':str(uuid5(NAMESPACE_DNS,s)),'symbol':s,'class':'us_equity','status':'active','exchange':exchange}

def volume_bars(p):
    result=bars(p)
    result['bars']={s:[dict(r,v=10000,vw=r['c']) for r in rs] for s,rs in result['bars'].items()}
    return result

class GlobalPreviewTests(unittest.TestCase):
    def test_turnover_uses_prior_fx(self):
        day=date(2026,11,1)
        sessions=[day+timedelta(days=i) for i in range(20)]
        values={d:(D(10),1000,D(10)) for d in sessions}
        rates={d-timedelta(days=1):D(4) for d in sessions}
        rates[sessions[-1]]=D(999)
        self.assertEqual(turnover(values,rates,sessions),D(40000))
        with self.assertRaises(ValueError):turnover(values,{},sessions)
    def test_preview(self):
        r=preview(asset,calendar,volume_bars,fx)
        self.assertEqual(len(r['instruments']),28)
        self.assertTrue(r['aligned_us_research'])
        self.assertFalse(r['eligible_for_orders'])
        self.assertEqual(len(r['plans']),5)
        for p in r['plans']:
            self.assertEqual(D(p['capital_pln']),D(50000))
            if p['account']=='E:global':
                self.assertFalse(p['allocations'])
                self.assertEqual(D(p['cash_pln']),D(50000))
            else:
                self.assertGreaterEqual(D(p['cash_pln']),D(10000))
                for a in p['allocations']:
                    self.assertLessEqual(D(a['notional_pln'])+D(a['estimated_entry_cost_pln']),D(2250))
                    self.assertNotIn(a['instrument_id'].split(':')[-1],('QQQ','TLT'))
    def test_missing_volume_blocks(self):
        with self.assertRaises(KeyError):preview(asset,calendar,bars,fx)
    def test_invalid_volume(self):
        from global_agent.alpaca_data import history
        d=date(2026,11,25)
        row={'t':'2026-11-25T05:00:00Z','o':'100','h':'100','l':'100','c':'100','v':-1,'vw':'100'}
        with self.assertRaises(ValueError):history(lambda p:{'bars':{'AAPL':[row]},'next_page_token':None},{('AAPL',d)},'raw',with_volume=True)
