import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal as D
from pathlib import Path
from global_agent.backtest import run, load_csv, signal


def sample(n=24):
    return [(date(2026,1,1)+timedelta(days=i), {'TEST': (D(100+i), D(100+i))}) for i in range(n)]

class BacktestTests(unittest.TestCase):
    def test_profiles(self):
        r=run(sample(), 'crypto', 'TEST')['profiles']
        self.assertEqual(set(r), set('ABCDE'))
        self.assertEqual(r['E']['fills'], 1)
        self.assertGreater(r['A']['fills'], 0)
        self.assertEqual(len(r['A']['history']), 4)
    def test_no_future_leak(self):
        bars=sample()
        first=run(bars[:22], 'crypto','TEST')['profiles']
        bars[23][1]['TEST']=(D(1),D(1))
        second=run(bars,'crypto','TEST')['profiles']
        for p in 'ABCDE':
            self.assertEqual(first[p]['history'], second[p]['history'][:2])
    def test_costs(self):
        a=run(sample(),'crypto','TEST',fee='0',spread='0',slippage='0')['profiles']
        b=run(sample(),'crypto','TEST',fee='.01',spread='.01',slippage='.01')['profiles']
        self.assertGreater(D(a['E']['equity_pln']),D(b['E']['equity_pln']))
    def test_csv_missing(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.csv'
            p.write_text('date,symbol,open_pln,close_pln\n2026-01-01,A,1,1\n2026-01-01,A,2,2\n')
            with self.assertRaises(ValueError):
                load_csv(p)
    def test_warmup(self):
        self.assertFalse(signal('A',[D(100)]*19,False))

if __name__=='__main__':
    unittest.main()
