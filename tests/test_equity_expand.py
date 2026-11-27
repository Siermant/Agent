import unittest
from uuid import uuid5,NAMESPACE_DNS
from urllib.error import HTTPError
from global_agent.equity_expand import discover,CANDIDATES

def asset(s):
    return {'id':str(uuid5(NAMESPACE_DNS,s)),'symbol':s,'class':'us_equity',
            'status':'active','exchange':'ARCA' if s=='SPY' else 'NASDAQ'}

class EquityExpandTests(unittest.TestCase):
    def test_discovery(self):
        u,e=discover(asset)
        self.assertEqual(len(CANDIDATES),30)
        self.assertEqual(len(u['XNAS']),29)
        self.assertEqual(e,[{'ticker':'SPY','reason':'venue_calendar_not_supported'}])
        self.assertEqual(u['XNAS']['QQQ'],'etf')
    def test_missing(self):
        def get(s):
            if s=='AAPL': raise HTTPError('https://example.invalid',404,'not found',None,None)
            return asset(s)
        u,e=discover(get)
        self.assertNotIn('AAPL',u['XNAS'])
        self.assertIn({'ticker':'AAPL','reason':'not_found'},e)
    def test_auth_failure(self):
        def get(s): raise HTTPError('https://example.invalid',403,'secret',None,None)
        with self.assertRaises(HTTPError): discover(get)
    def test_bad_identity(self):
        with self.assertRaises(ValueError): discover(lambda s:dict(asset(s),symbol='WRONG'))
