import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from global_agent.history import candle_page, prior_rate, download


def candle(day, confirm='1'):
    stamp=int(datetime.combine(day,datetime.min.time(),timezone.utc).timestamp()*1000)
    return [str(stamp),'10','12','9','11','1','1','1',confirm]

class HistoryTests(unittest.TestCase):
    def test_closed_only(self):
        day=date(2025,1,1)
        rows,_=candle_page({'code':'0','data':[candle(day),candle(day+timedelta(days=1),'0')]})
        self.assertEqual(rows,{day:(D(10),D(11))})
    def test_prior_fx(self):
        day=date(2025,1,3)
        rates={day-timedelta(days=1):D('4.2'),day:D('9')}
        self.assertEqual(prior_rate(day,rates,sorted(rates))[1],D('4.2'))
        with self.assertRaises(ValueError):
            prior_rate(day+timedelta(days=10),rates,sorted(rates))
    def test_export_and_missing(self):
        start=date(2025,1,1)
        end=start+timedelta(days=20)
        days=[start+timedelta(days=i) for i in range(21)]
        def request(url):
            if 'nbp.pl' in url:
                return {'code':'EUR','rates':[{'effectiveDate':'2024-12-31','mid':'4.2'}]+[{'effectiveDate':d.isoformat(),'mid':'4.3'} for d in days]}
            return {'code':'0','data':[candle(d) for d in reversed(days)]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prices.csv'
            self.assertEqual(download(start,end,['BTC-EUR'],path,request),21)
            original=path.read_text()
            self.assertIn('2025-01-01,BTC-EUR,42.0,46.2,10,11,2024-12-31,4.2',original)
            def missing(url):
                if 'nbp.pl' in url:
                    return request(url)
                return {'code':'0','data':[]}
            with self.assertRaises(ValueError):
                download(start,end,['BTC-EUR'],path,missing)
            self.assertEqual(path.read_text(),original)

if __name__=='__main__':
    unittest.main()
