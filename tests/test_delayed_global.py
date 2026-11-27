import unittest,tempfile
from pathlib import Path
from datetime import timedelta
from global_agent.equity_quotes import qualify
from global_agent.global_paper_initial import open_initial
from global_agent.global_paper_mark import mark
from global_agent.global_panel import export_panel,summary,render_global
from test_global_paper_initial import NOW,plan,snapshot
from test_equity_quotes import clock,quotes,fx

def delayed(at=NOW):
    s=snapshot()
    s.update(source='alpaca_delayed_sip_latest',quote_delay_seconds=900,
             checked_utc=at.isoformat(),session_close_utc=(at+timedelta(hours=2)).isoformat())
    for q in s['quotes'].values():q['observed_at']=(at-timedelta(seconds=905)).isoformat()
    return s

class DelayedTests(unittest.TestCase):
    def test_age_window(self):
        q=quotes()
        for x in q['quotes'].values():x['t']=(NOW-timedelta(seconds=905)).isoformat()
        r=qualify(clock,lambda:q,fx,lambda:NOW,feed='delayed_sip',include_quotes=True)
        self.assertTrue(r['ready'])
        self.assertEqual(r['quote_delay_seconds'],900)
        self.assertFalse(qualify(clock,quotes,fx,lambda:NOW,feed='delayed_sip')['ready'])
        self.assertFalse(qualify(clock,lambda:q,fx,lambda:NOW)['ready'])
    def test_open_mark_panel_same_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            r=open_initial(root,plan(),delayed(),lambda:NOW)
            self.assertEqual(r['quote_delay_seconds'],900)
            self.assertEqual(len(r['fills']),4)
            later=NOW+timedelta(minutes=5)
            def fetch(symbols):
                s=delayed(later)
                s['quotes']={k:s['quotes'][k] for k in symbols}
                return s
            v=mark(root,fetch,lambda:later)
            self.assertEqual(v['quote_source'],r['quote_source'])
            export_panel(root,v,v)
            out=summary(root,later)
            self.assertEqual(out['state'],'valued')
            self.assertEqual(out['quote_delay_seconds'],900)
            self.assertIn('15 minut',render_global(out))
            def wrong(symbols):
                s=fetch(symbols);s.update(source='alpaca_iex_latest',quote_delay_seconds=0)
                return s
            with self.assertRaises(ValueError):mark(root,wrong,lambda:later)
    def test_no_partial_delayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):open_initial(tmp,plan(),delayed(),lambda:NOW,allow_unquoted_cash=True)
            self.assertFalse((Path(tmp)/'global-paper.db').exists())
