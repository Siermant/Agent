import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime,timezone
from urllib.error import HTTPError
from global_agent.equity_job import cycle, credentials

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

def result():
    return {'research_only':True,'real_trading':False,'eligible_for_orders':False,
            'state':'ranked','expected_instruments':1,'generated_utc':NOW.isoformat(),
            'cohorts':[{'instruments':[{'ticker':'TEST'}]}]}

class EquityJobTests(unittest.TestCase):
    def test_success_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def fetch():
                self.assertEqual(json.loads((p/'status.json').read_text())['state'],'running')
                return result()
            self.assertEqual(cycle(p,fetch,lambda:NOW)['state'],'ok')
            old=(p/'ranking.json').read_bytes()
            def denied():
                raise HTTPError('https://example.invalid',403,'SECRET',None,None)
            s=cycle(p,denied,lambda:NOW)
            self.assertEqual(s['state'],'failed')
            self.assertEqual(s['http_status'],403)
            self.assertNotIn('SECRET',str(s))
            self.assertEqual((p/'ranking.json').read_bytes(),old)
    def test_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            s=cycle(p,lambda:dict(result(),expected_instruments=2),lambda:NOW)
            self.assertEqual(s['state'],'failed')
            self.assertFalse((p/'ranking.json').exists())
    def test_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'alpaca-paper.json'
            p.write_text(json.dumps({'key':'dummykey','secret':'dummysecret'}))
            self.assertEqual(credentials(tmp),('dummykey','dummysecret'))
            p.write_text(json.dumps({'key':'dummykey','secret':'bad value'}))
            with self.assertRaises(ValueError): credentials(tmp)
