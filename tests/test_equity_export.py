import unittest
import tempfile
import json
from datetime import datetime,timezone
from pathlib import Path
from global_agent.equity_export import export
from global_agent.equity_panel import summary

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)

class EquityExportTests(unittest.TestCase):
    def test_allowlist(self):
        flags={'research_only':True,'real_trading':False,'eligible_for_orders':False}
        s=dict(flags,state='ok',finished_utc=NOW.isoformat(),ranking_generated_utc=NOW.isoformat(),instruments=1,cohorts=1,secret='DO_NOT_EXPORT')
        r=dict(flags,state='ranked',generated_utc=NOW.isoformat(),expected_instruments=1,secret='DO_NOT_EXPORT',cohorts=[
            dict(mic='XNAS',source='alpaca_iex_daily',last_session='2026-11-25',instruments=[
                dict(instrument_id='test',ticker='TEST',mic='XNAS',rank=1,score=3,return_20_pct=1,return_60_pct=2,annualized_volatility_pct=3,max_drawdown_60_pct=4,secret='DO_NOT_EXPORT')])])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'panel.json'
            export(tmp,s,r)
            self.assertNotIn('DO_NOT_EXPORT',p.read_text())
            self.assertEqual(p.stat().st_mode & 0o777,0o644)
            self.assertEqual(summary(tmp,NOW)['state'],'ok')
            export(tmp,dict(flags,state='failed'))
            self.assertNotIn('ranking',json.loads(p.read_text()))
            self.assertEqual(summary(tmp,NOW)['state'],'scan_not_ok')
