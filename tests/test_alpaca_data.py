import unittest
from datetime import date
from global_agent.alpaca_data import history

DAY=date(2026,11,25)
ROW={'t':'2026-11-25T05:00:00Z','o':'99','h':'102','l':'98','c':'100'}

class AlpacaDataTests(unittest.TestCase):
    def test_pages(self):
        calls=[]
        def request(p):
            calls.append(p)
            return {'bars':{'AAPL' if len(calls)==1 else 'MSFT':[ROW]},'next_page_token':'next' if len(calls)==1 else None}
        result=history(request,{('AAPL',DAY),('MSFT',DAY)},'split,dividend')
        self.assertEqual(len(result),2)
        self.assertEqual(calls[0]['feed'],'iex')
        self.assertEqual(calls[1]['page_token'],'next')
        self.assertEqual(calls[0]['adjustment'],'split,dividend')
    def test_missing(self):
        with self.assertRaisesRegex(ValueError,'Missing requested'):
            history(lambda p:{'bars':{'AAPL':[ROW]},'next_page_token':None},{('AAPL',DAY),('MSFT',DAY)},'raw')
    def test_wrong_time(self):
        with self.assertRaises(ValueError):
            history(lambda p:{'bars':{'AAPL':[dict(ROW,t='2026-11-25T06:00:00Z')]},'next_page_token':None},{('AAPL',DAY)},'raw')
    def test_duplicate(self):
        with self.assertRaises(ValueError):
            history(lambda p:{'bars':{'AAPL':[ROW,ROW]},'next_page_token':None},{('AAPL',DAY)},'raw')
    def test_invalid_ohlc(self):
        with self.assertRaises(ValueError):
            history(lambda p:{'bars':{'AAPL':[dict(ROW,c='150')]},'next_page_token':None},{('AAPL',DAY)},'raw')
