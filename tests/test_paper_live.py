import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from decimal import Decimal as D
from global_agent.ledger import Ledger
from global_agent.market import QuoteStore
from global_agent.paper_live import PaperRunner, UNIVERSE

class PaperLiveTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.now = datetime(2026,11,26,12,tzinfo=timezone.utc)
        self.ledger = Ledger(Path(tmp.name)/'ledger.db',clock=lambda:self.now)
        self.store = QuoteStore(Path(tmp.name)/'market.db')
        self.trend = [D(100+i*2) for i in range(20)]
        self.runner = PaperRunner(self.ledger,self.store,self.context)
    def context(self,day):
        return day-timedelta(days=1),D(4),{s:self.trend[:] for s in UNIVERSE}
    def feed(self, symbols=UNIVERSE):
        self.store.record([(s,self.now.isoformat(timespec='milliseconds'),'99','100','99.5') for s in symbols],self.now)
    def fills(self):
        return sum(s['fills'] for s in self.ledger.status())
    def test_restart_no_duplicate(self):
        self.feed()
        self.assertEqual(self.runner.tick(),'ok')
        count=self.fills()
        self.assertGreater(count,1)
        self.runner=PaperRunner(self.ledger,self.store,self.context)
        self.assertEqual(self.runner.tick(),'ok')
        self.assertEqual(self.fills(),count)
        self.assertTrue(all(not p['fills'] for p in self.ledger.status() if not p['account'].endswith(':crypto')))
    def test_outage_and_recovery(self):
        self.feed()
        self.store.fail(self.now,'offline')
        self.assertEqual(self.runner.tick(),'feed_unavailable')
        self.assertEqual(self.fills(),0)
        self.feed()
        self.assertEqual(self.runner.tick(),'ok')
        count=self.fills()
        self.now+=timedelta(minutes=2)
        self.assertEqual(self.runner.tick(),'feed_unavailable')
        self.assertEqual(self.fills(),count)
    def test_missing_quote(self):
        self.feed(('BTC-EUR',))
        self.assertEqual(self.runner.tick(),'incomplete_quotes')
        self.assertEqual(self.fills(),0)
    def test_recover_after_fill_before_decision(self):
        self.feed()
        self.runner.tick()
        count=self.fills()
        with self.ledger.tx() as db:
            db.execute('DELETE FROM paper_decisions')
        self.runner=PaperRunner(self.ledger,self.store,self.context)
        self.runner.tick()
        self.assertEqual(self.fills(),count)
    def test_next_day_sell(self):
        self.feed()
        self.runner.tick()
        self.now+=timedelta(days=1)
        self.trend=[D(200-i*4) for i in range(20)]
        self.feed()
        self.runner.tick()
        self.assertEqual(self.runner.state('A:crypto')['positions'],{})
        self.assertEqual(self.runner.state('E:crypto')['fills'],1)
    def test_stale_fx(self):
        runner=PaperRunner(self.ledger,self.store,lambda d:(d,D(4),{s:self.trend for s in UNIVERSE}))
        self.feed()
        with self.assertRaises(ValueError): runner.tick()
        self.assertEqual(self.fills(),0)
