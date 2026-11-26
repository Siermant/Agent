import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from global_agent.ledger import Ledger
from global_agent.market import QuoteStore
from global_agent.diagnostics import statistics, report
from global_agent.backup import daily_backup

class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.data=Path(t.name)
        self.now=datetime(2026,11,26,12,tzinfo=timezone.utc)
        self.ledger=Ledger(self.data/'ledger.db',clock=lambda:self.now)
        self.store=QuoteStore(self.data/'market.db')
    def buy(self):
        self.ledger.execute('A:crypto','test','BTC-EUR','buy','1',{'BTC-EUR':('99','100',self.now)},'.01',market_open=True,approved=True)
    def test_unknown_not_zero(self):
        rows=statistics(self.ledger,self.now)
        self.assertEqual(len(rows),15)
        self.assertTrue(all(r['return_pct'] is None and not r['fresh'] for r in rows))
    def test_costs_and_stale(self):
        self.buy()
        r=next(r for r in statistics(self.ledger,self.now) if r['account']=='A:crypto')
        self.assertEqual(D(r['fees_pln']),D(1))
        self.assertEqual(D(r['equity_pln']),D(19998))
        self.assertEqual(D(r['return_pct']),D('-.01'))
        self.assertTrue(r['fresh'])
        self.assertFalse(next(r for r in statistics(self.ledger,self.now+timedelta(seconds=91)) if r['account']=='A:crypto')['fresh'])
    def test_report_allowlist(self):
        self.store.fail(self.now,'SECRET_TEST_NOT_EXPORTED')
        state={'version':'test','mode':'monitor-only','paper':{'enabled':False,'state':'disabled'},'secret':'SECRET_TEST_NOT_EXPORTED'}
        with ZipFile(BytesIO(report(self.ledger,self.store,state,self.now))) as z:
            self.assertEqual(set(z.namelist()),{'status.json','portfolios.json','decisions.json'})
            self.assertNotIn(b'SECRET_TEST_NOT_EXPORTED',b''.join(z.read(n) for n in z.namelist()))
            self.assertEqual(len(json.loads(z.read('portfolios.json'))),15)
    def test_backup_restore_and_retention(self):
        self.buy()
        first=daily_backup(self.data,self.now,keep=2)
        self.assertEqual(first,daily_backup(self.data,self.now,keep=2))
        restored=Ledger(first/'ledger.db',clock=lambda:self.now)
        self.assertEqual(next(r for r in restored.status() if r['account']=='A:crypto')['fills'],1)
        daily_backup(self.data,self.now+timedelta(days=1),keep=2)
        last=daily_backup(self.data,self.now+timedelta(days=2),keep=2)
        self.assertFalse(first.exists())
        self.assertTrue((last/'market.db').exists())
    def test_failed_backup_not_published(self):
        (self.data/'market.db').unlink()
        with self.assertRaises(Exception): daily_backup(self.data,self.now)
        self.assertFalse((self.data/'backups'/self.now.date().isoformat()).exists())
