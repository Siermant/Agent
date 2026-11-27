import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from decimal import Decimal as D
from global_agent.ledger import Ledger
from global_agent.market import QuoteStore
from global_agent.paper_live import PaperRunner, UNIVERSE, ALL_UNIVERSE, configured_universe
from unittest.mock import patch
import os

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

    def all_context(self, day):
        return day-timedelta(days=1),D(4),{s:self.trend[:] for s in ALL_UNIVERSE}

    def test_all_pairs_and_caps(self):
        self.feed(ALL_UNIVERSE)
        runner=PaperRunner(self.ledger,self.store,self.all_context,ALL_UNIVERSE)
        self.assertEqual(runner.tick(),'ok')
        for p in 'ABCD':
            state=runner.state(p+':crypto')
            self.assertEqual(set(state['positions']),set(ALL_UNIVERSE))
            equity=D(state['last_equity_pln'])
            self.assertGreaterEqual(D(state['cash_pln']),equity*D('.20'))
            for qty in state['positions'].values():
                self.assertLessEqual(D(qty)*D(396),equity*D('.05'))
        self.assertEqual(set(runner.state('E:crypto')['positions']),{'BTC-EUR'})
        count=self.fills()
        self.assertEqual(runner.tick(),'ok')
        self.assertEqual(count,self.fills())

    def test_core_then_all_same_day(self):
        self.feed()
        self.assertEqual(self.runner.tick(),'ok')
        count=self.fills()
        self.feed(ALL_UNIVERSE)
        extended=PaperRunner(self.ledger,self.store,self.all_context,ALL_UNIVERSE)
        self.assertEqual(extended.tick(),'ok')
        self.assertEqual(self.fills(),count+24)
        self.assertEqual(self.runner.tick(),'incompatible_positions')
        self.assertEqual(self.fills(),count+24)

    def test_all_missing_quote_blocks(self):
        self.feed(ALL_UNIVERSE[:-1])
        runner=PaperRunner(self.ledger,self.store,self.all_context,ALL_UNIVERSE)
        self.assertEqual(runner.tick(),'incomplete_quotes')
        self.assertEqual(self.fills(),0)

    def test_universe_config(self):
        for mode,expected in [('core',UNIVERSE),('all',ALL_UNIVERSE)]:
            with patch.dict(os.environ,{'GTA_CRYPTO_UNIVERSE':mode}):
                self.assertEqual(configured_universe(),expected)
        with patch.dict(os.environ,{'GTA_CRYPTO_UNIVERSE':'typo'}):
            with self.assertRaises(ValueError):
                configured_universe()
