import tempfile
import unittest
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
from global_agent.fx_reference import rates, export


class FXReferenceTests(unittest.TestCase):
    def test_chunks(self):
        calls=[]
        def request(url):
            parts=urlsplit(url).path.strip('/').split('/')
            c,start,end=parts[-3:]
            calls.append((date.fromisoformat(start),date.fromisoformat(end)))
            return {'table':'A','code':c.upper(),'rates':[{'effectiveDate':start,'mid':'4.20'}]}
        result=rates('USD',date(2026,1,1),date(2026,5,1),request)
        self.assertEqual(len(calls),2)
        self.assertEqual(len(result),2)
        self.assertTrue(all((b-a).days<=89 for a,b in calls))
    def test_wrong_currency(self):
        with self.assertRaises(ValueError):
            rates('USD',date(2026,1,1),date(2026,1,2),lambda _: {'table':'A','code':'EUR','rates':[]})
    def test_duplicate(self):
        row={'effectiveDate':'2026-01-01','mid':'4'}
        with self.assertRaises(ValueError):
            rates('USD',date(2026,1,1),date(2026,1,2),lambda _: {'table':'A','code':'USD','rates':[row,row]})
    def test_atomic_export(self):
        def request(url):
            c=urlsplit(url).path.strip('/').split('/')[-3].upper()
            return {'table':'A','code':c,'rates':[{'effectiveDate':'2026-11-24','mid':'4.2'},{'effectiveDate':'2026-11-25','mid':'4.1'}]}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fx.csv'
            self.assertEqual(export(date(2026,11,25),date(2026,11,26),p,request,date(2026,11,27)),4)
            old=p.read_text()
            with self.assertRaises(ValueError):
                export(date(2026,11,25),date(2026,11,26),p,lambda _: {},date(2026,11,27))
            self.assertEqual(p.read_text(),old)
