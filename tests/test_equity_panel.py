import unittest
import tempfile
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
from global_agent.equity_panel import summary,render_equities

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

class EquityPanelTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        flags={'research_only':True,'real_trading':False,'eligible_for_orders':False}
        stamp=(NOW-timedelta(minutes=1)).isoformat()
        self.s=dict(flags,state='ok',finished_utc=NOW.isoformat(),ranking_generated_utc=stamp,instruments=1,cohorts=1)
        row=dict(instrument_id='test',mic='XNAS',ticker='<script>',rank=1,
                 return_20_pct=1,return_60_pct=2,annualized_volatility_pct=3,max_drawdown_60_pct=4,score=5)
        self.r=dict(flags,state='ranked',generated_utc=stamp,expected_instruments=1,
                    cohorts=[dict(mic='XNAS',source='alpaca_iex_daily',last_session='2026-11-25',instruments=[row])])
    def read(self,now=NOW):
        for name,obj in [('status',self.s),('ranking',self.r)]:
            (self.root/(name+'.json')).write_text(json.dumps(obj))
        return summary(self.root,now)
    def test_valid_and_escape(self):
        r=self.read()
        self.assertEqual(r['state'],'ok')
        html=render_equities(r)
        self.assertIn('&lt;script&gt;',html)
        self.assertNotIn('<script>',html)
    def test_stale(self):
        self.assertEqual(self.read(NOW+timedelta(days=2))['state'],'stale')
    def test_incomplete_or_failed(self):
        self.s['state']='failed'
        self.assertEqual(self.read()['state'],'scan_not_ok')
        self.s['state']='ok'
        self.s['instruments']=2
        self.assertEqual(self.read()['state'],'unavailable')
    def test_mixed_reports(self):
        self.s['ranking_generated_utc']=NOW.isoformat()
        self.assertEqual(self.read()['state'],'unavailable')
    def test_no_connection(self):
        self.assertEqual(summary(None)['state'],'not_connected')
    def test_real_trading_flag(self):
        self.r['real_trading']=True
        self.assertEqual(self.read()['state'],'unavailable')
