import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from global_agent.ledger import Ledger


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'ledger.db'
        self.now = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        self.ledger = Ledger(self.path, clock=lambda: self.now)

    def quotes(self, bid='99', ask='100'):
        return {'TEST': (bid, ask, self.now)}

    def fill(self, order='1', side='buy', qty='10', account='A:global', **kw):
        return self.ledger.execute(account, order, 'TEST', side, qty, kw.pop('quotes', self.quotes()), kw.pop('fee_rate', '.001'), market_open=kw.pop('market_open', True), approved=kw.pop('approved', True), **kw)

    def state(self, account='A:global'):
        return next(r for r in self.ledger.status() if r['account'] == account)

    def test_initial(self):
        self.assertEqual(len(self.ledger.status()), 15)
        self.assertEqual(sum(D(r['cash_pln']) for r in self.ledger.status()), D('491300'))

    def test_persistence_isolation_duplicate(self):
        self.assertEqual(self.fill()['fee_pln'], '1.00')
        self.ledger = Ledger(self.path, clock=lambda: self.now)
        self.assertEqual(D(self.state()['cash_pln']), D('48999'))
        self.assertEqual(D(self.state('B:global')['cash_pln']), D('50000'))
        with self.assertRaises(ValueError):
            self.fill()
        self.assertEqual(self.state()['fills'], 1)

    def test_costs_and_sell(self):
        self.fill(slippage='.01', min_fee='2')
        self.assertEqual(D(self.state()['cash_pln']), D('48988'))
        self.fill(order='2', side='sell')
        self.assertEqual(D(self.state()['cash_pln']), D('49977.01'))
        self.assertEqual(self.state()['positions'], {})

    def test_block_invalid(self):
        for opts in ({'qty':'100'}, {'side':'sell'}, {'approved':False}, {'market_open':False}, {'qty':'NaN'}, {'quotes':{'TEST':('99','100',self.now-timedelta(minutes=2))}}):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                self.fill(**opts)
        self.assertEqual(self.state()['fills'], 0)
        self.assertEqual(D(self.state()['cash_pln']), D('50000'))

    def test_loss_gate_and_reduction(self):
        self.fill(qty='24')
        self.now += timedelta(seconds=1)
        quotes = self.quotes('50','51')
        self.assertTrue(self.ledger.mark('A:global', quotes)['risk_blocked'])
        with self.assertRaises(ValueError):
            self.fill(order='2', qty='1', quotes=quotes)
        self.fill(order='3', side='sell', qty='24', quotes=quotes)
        self.assertEqual(self.state()['positions'], {})

    def test_benchmark(self):
        self.fill(account='E:global', qty='490')
        with self.assertRaises(ValueError):
            self.fill(account='E:global', order='2', qty='1')


if __name__ == '__main__':
    unittest.main()
