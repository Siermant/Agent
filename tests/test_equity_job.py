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


class ExpandedJobTests(unittest.TestCase):
    def setup_scan(self):
        from global_agent.equity_job import QUALIFIED, expanded_scan
        from global_agent.equity_expand import CANDIDATES
        from uuid import uuid5, NAMESPACE_DNS
        def asset(symbol):
            exchange = ('NASDAQ' if symbol in QUALIFIED['XNAS'] else
                        'NYSE' if symbol in QUALIFIED['XNYS'] else 'ARCA')
            return {'id':str(uuid5(NAMESPACE_DNS,symbol)), 'symbol':symbol,
                    'class':'us_equity','status':'active','exchange':exchange}
        def scanner(*args,universe):
            return dict(result(),expected_instruments=28,
                        cohorts=[{'mic':mic,'instruments':[
                            {'ticker':s,'mic':mic,'kind':kind}
                            for s,kind in items.items()]} for mic,items in universe.items()])
        return expanded_scan,asset,scanner
    def test_exact_membership(self):
        expand,asset,scanner=self.setup_scan()
        r=expand(asset,None,None,scanner)
        self.assertEqual(r['expected_instruments'],28)
        self.assertEqual(r['candidate_count'],30)
        self.assertEqual({x['ticker'] for x in r['discovery_excluded']},{'SPY','VT'})
    def test_exchange_change_blocks(self):
        expand,asset,scanner=self.setup_scan()
        with self.assertRaisesRegex(ValueError,'Qualified universe changed'):
            expand(lambda s:dict(asset(s),exchange='NYSE') if s=='WMT' else asset(s),
                   None,None,scanner)
    def test_missing_rank_blocks(self):
        expand,asset,scanner=self.setup_scan()
        def incomplete(*args,**kwargs):
            r=scanner(*args,**kwargs)
            r['cohorts'][0]['instruments'].pop()
            return r
        with self.assertRaisesRegex(ValueError,'Incomplete qualified ranking'):
            expand(asset,None,None,incomplete)


class PanelExportCycleTests(unittest.TestCase):
    def test_cycle_exports_success_then_failure(self):
        from global_agent.equity_panel import summary
        with tempfile.TemporaryDirectory() as state, tempfile.TemporaryDirectory() as panel:
            p=Path(panel)/'panel.json'
            def fetch():
                self.assertEqual(json.loads(p.read_text())['status']['state'],'running')
                row=dict(instrument_id='fixture',ticker='TEST',mic='XNAS',rank=1,
                         return_20_pct=1,return_60_pct=2,annualized_volatility_pct=3,
                         max_drawdown_60_pct=4,score=5)
                return dict(result(),cohorts=[dict(mic='XNAS',source='alpaca_iex_daily',
                    last_session='2026-11-25',instruments=[row])])
            self.assertEqual(cycle(state,fetch,lambda:NOW,export_dir=panel)['state'],'ok')
            self.assertEqual(summary(panel,NOW)['state'],'ok')
            old=(Path(state)/'ranking.json').read_bytes()
            def denied():
                raise RuntimeError('SECRET')
            self.assertEqual(cycle(state,denied,lambda:NOW,export_dir=panel)['state'],'failed')
            self.assertEqual(summary(panel,NOW)['state'],'scan_not_ok')
            self.assertEqual((Path(state)/'ranking.json').read_bytes(),old)
            self.assertNotIn('SECRET',p.read_text())

    def test_bad_export_dir_fails_closed(self):
        with tempfile.TemporaryDirectory() as state:
            called=[]
            s=cycle(state,lambda:called.append(True),lambda:NOW,
                    export_dir=Path(state)/'missing')
            self.assertEqual(s['state'],'failed')
            self.assertEqual(called,[])
