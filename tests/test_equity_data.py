import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from global_agent.equity_data import load, FIELDS, CAL_FIELDS, FX_FIELDS


class EquityDataTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.now=datetime(2026,11,27,12,tzinfo=timezone.utc)
        self.calendar=[['XNYS','2026-11-25','2026-11-25T21:00:00Z']]
        self.bars=[['test:US1','TEST','XNYS','stock','USD','America/New_York','2026-11-25',
                    '2026-11-25T21:00:00Z','2026-11-25T22:00:00Z','100','90','split_dividend','fixture']]
        self.fx=[['USD','2026-11-24','4'],['USD','2026-11-25','9']]

    def run_load(self):
        paths=[]
        for name,fields,data in [('calendar',CAL_FIELDS,self.calendar),('bars',FIELDS,self.bars),('fx',FX_FIELDS,self.fx)]:
            path=self.root/(name+'.csv')
            with path.open('w',newline='') as f:
                writer=csv.writer(f)
                writer.writerow(fields)
                writer.writerows(data)
            paths.append(path)
        return load(*paths,now=self.now)

    def test_prior_fx_and_adjustment(self):
        result=self.run_load()
        row=result['bars'][0]
        self.assertEqual(row['raw_close_pln'],'400')
        self.assertEqual(row['adjusted_close_pln'],'360')
        self.assertEqual(row['fx_date'],'2026-11-24')
        self.assertFalse(result['calendar_verified_externally'])
        self.assertFalse(result['real_trading'])

    def test_missing_session(self):
        self.calendar.append(['XNYS','2026-11-24','2026-11-24T21:00:00Z'])
        with self.assertRaisesRegex(ValueError,'Missing supplied'):
            self.run_load()

    def test_duplicate_bar(self):
        self.bars.append(self.bars[0][:])
        with self.assertRaises(ValueError): self.run_load()

    def test_future_retrieval(self):
        self.bars[0][8]='2026-11-28T00:00:00Z'
        with self.assertRaises(ValueError): self.run_load()

    def test_wrong_close(self):
        self.bars[0][7]='2026-11-25T20:00:00Z'
        with self.assertRaises(ValueError): self.run_load()

    def test_stale_fx(self):
        self.fx=[['USD','2026-11-01','4']]
        with self.assertRaisesRegex(ValueError,'prior reference FX'): self.run_load()

    def test_raw_only(self):
        self.bars[0][10:12]=['','none']
        self.assertIsNone(self.run_load()['bars'][0]['adjusted_close_pln'])

    def test_invalid_price(self):
        self.bars[0][9]='NaN'
        with self.assertRaises(ValueError): self.run_load()

    def test_early_close_and_pln(self):
        self.calendar=[['XWAR','2026-11-25','2026-11-25T12:00:00Z']]
        self.bars=[['test:PL1','TEST','XWAR','etf','PLN','Europe/Warsaw','2026-11-25',
                    '2026-11-25T12:00:00Z','2026-11-25T13:00:00Z','100','','none','fixture']]
        self.fx=[]
        row=self.run_load()['bars'][0]
        self.assertEqual(row['raw_close_pln'],'100')
        self.assertIsNone(row['fx_date'])
