import unittest
from datetime import datetime,timedelta,timezone
from global_agent.equity_quotes import qualify,SYMBOLS
NOW=datetime(2026,11,27,15,tzinfo=timezone.utc)
def clock(opened=True):
    return {'timestamp':NOW.isoformat(),'is_open':opened,'next_open':(NOW+timedelta(days=3)).isoformat(),'next_close':(NOW+timedelta(hours=2)).isoformat()}
def quotes():
    return {'quotes':{s:{'t':(NOW-timedelta(seconds=5)).isoformat(),'bp':'100','ap':'100.1','bs':1,'as':1} for s in SYMBOLS}}
def fx(_):
    return {'table':'A','code':'USD','rates':[{'effectiveDate':'2026-11-26','mid':'4'}]}
class EquityQuotesTests(unittest.TestCase):
    def test_open(self):
        r=qualify(clock,quotes,fx,lambda:NOW)
        self.assertTrue(r['ready'])
        self.assertEqual(r['valid_quotes'],26)
        self.assertFalse(r['eligible_for_orders'])
        self.assertNotIn('quotes',r)
    def test_closed(self):
        r=qualify(lambda:clock(False),lambda:self.fail('no quotes'),lambda _:self.fail('no fx'),lambda:NOW)
        self.assertEqual(r['state'],'market_closed')
        self.assertFalse(r['ready'])
    def test_bad_quote(self):
        q=quotes()
        q['quotes']['WMT']['t']=(NOW-timedelta(seconds=61)).isoformat()
        q['quotes']['AAPL']['ap']='102'
        r=qualify(clock,lambda:q,fx,lambda:NOW)
        self.assertFalse(r['ready'])
        self.assertEqual(set(r['rejected_symbols']),{'WMT','AAPL'})
    def test_session_changes(self):
        calls=iter([clock(),clock(False)])
        r=qualify(lambda:next(calls),quotes,fx,lambda:NOW)
        self.assertEqual(r['state'],'session_changed')
    def test_stale_clock(self):
        with self.assertRaises(ValueError):qualify(lambda:dict(clock(),timestamp=(NOW-timedelta(minutes=1)).isoformat()),quotes,fx,lambda:NOW)
