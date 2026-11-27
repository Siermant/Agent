import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from datetime import date
from global_agent.alpaca_data import history, collect, write_csv
from global_agent.equity_data import load

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

    def test_collect_and_import(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            manifest=root/'manifest.csv'
            calendar=root/'calendar.csv'
            bars=root/'bars.csv'
            fx=root/'fx.csv'
            manifest.write_text('instrument_id,ticker,mic,kind\nlisting:1,AAPL,XNAS,stock\n')
            calendar.write_text('mic,session,close_at\nXNAS,2026-11-25,2026-11-25T21:00:00Z\n')
            fx.write_text('currency,date,pln_per_unit\nUSD,2026-11-24,4\n')
            def request(p):
                row=dict(ROW) if p['adjustment']=='raw' else dict(ROW,o='89',h='92',l='88',c='90')
                return {'bars':{'AAPL':[row]},'next_page_token':None}
            data=collect(manifest,calendar,request,datetime(2026,11,27,12,tzinfo=timezone.utc))
            write_csv(bars,data)
            result=load(calendar,bars,fx,now=max(datetime(2026,11,27,12,tzinfo=timezone.utc),datetime.now(timezone.utc)))
            self.assertEqual(result['bars'][0]['raw_close_pln'],'400')
            self.assertEqual(result['bars'][0]['adjusted_close_pln'],'360')
            self.assertFalse(result['real_trading'])

    def test_current_session_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            m=root/'m.csv'
            c=root/'c.csv'
            m.write_text('instrument_id,ticker,mic,kind\nlisting:1,AAPL,XNAS,stock\n')
            c.write_text('mic,session,close_at\nXNAS,2026-11-25,2026-11-25T21:00:00Z\n')
            with self.assertRaisesRegex(ValueError,'completed US calendar'):
                collect(m,c,lambda _: self.fail('No network expected'),datetime(2026,11,25,23,tzinfo=timezone.utc))
