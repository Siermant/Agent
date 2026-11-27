import unittest
import tempfile
import json
from datetime import timedelta
from pathlib import Path
from global_agent.global_panel import export_panel,summary,render_global
from global_agent.global_paper_initial import open_initial
from global_agent.global_paper_mark import mark
from test_global_paper_initial import plan,snapshot,NOW
from test_global_paper_mark import new_snapshot

class GlobalPanelTests(unittest.TestCase):
    def test_export_render_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            open_initial(root,plan(),snapshot(),lambda:NOW)
            now=NOW+timedelta(minutes=5)
            v=mark(root,lambda _:new_snapshot(),lambda:now)
            v['secret']='do-not-publish'
            export_panel(root,v,v)
            self.assertNotIn('do-not-publish',(root/'global.json').read_text())
            s=summary(root,now)
            self.assertEqual(s['state'],'valued')
            self.assertTrue(s['fresh'])
            self.assertEqual(len(s['accounts']),5)
            self.assertEqual(len(s['fills']),4)
            html=render_global(s)
            self.assertIn('WMT',html)
            self.assertIn('<svg',html)
            self.assertFalse(summary(root,now+timedelta(minutes=15))['fresh'])
            self.assertEqual(summary(root,now+timedelta(days=8))['state'],'stale')
            closed=dict(v,state='market_closed',checked_utc=(now+timedelta(hours=3)).isoformat())
            export_panel(root,closed,v)
            s=summary(root,now+timedelta(hours=3))
            self.assertEqual(s['state'],'market_closed')
            self.assertFalse(s['fresh'])
            self.assertEqual(len(s['accounts']),5)
    def test_invalid_flags_and_ticker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            open_initial(root,plan(),snapshot(),lambda:NOW)
            now=NOW+timedelta(minutes=5)
            v=mark(root,lambda _:new_snapshot(),lambda:now)
            export_panel(root,dict(v,real_trading=True),v)
            self.assertEqual(summary(root,now)['state'],'unavailable')
            v['accounts'][0]['positions'][0]['ticker']='<script>'
            export_panel(root,v,v)
            self.assertEqual(summary(root,now)['state'],'unavailable')
    def test_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            v=mark(tmp,lambda _:self.fail('unexpected'),lambda:NOW)
            export_panel(tmp,v)
            self.assertEqual(summary(tmp,NOW)['state'],'waiting_for_initial_open')
