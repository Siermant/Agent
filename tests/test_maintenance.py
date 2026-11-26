import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from global_agent.ledger import Ledger
from global_agent.maintenance import compact_snapshots
from global_agent.service_guard import service_lock
from global_agent.diagnostics import recent_executions

class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        self.root=Path(t.name)
        self.now=datetime(2026,11,1,12,tzinfo=timezone.utc)
        self.ledger=Ledger(self.root/'ledger.db',clock=lambda:self.now)
    def test_lock_released_and_second_instance_blocked(self):
        with service_lock(self.root):
            with self.assertRaises(RuntimeError):
                with service_lock(self.root): pass
        with service_lock(self.root): pass
    def test_compact_keeps_daily_last_and_peak(self):
        with self.ledger.tx() as db:
            for day in (1,2,25):
                for hour,eq in ((10,'21000'),(11,'20000')):
                    db.execute('INSERT INTO snapshots VALUES(?,?,?)',('A:crypto',datetime(2026,11,day,hour,tzinfo=timezone.utc).isoformat(timespec='microseconds'),eq))
            db.execute('UPDATE accounts SET peak=? WHERE id=?',('21000','A:crypto'))
        now=datetime(2026,11,26,12,tzinfo=timezone.utc)
        self.assertEqual(compact_snapshots(self.ledger,now),2)
        self.assertEqual(compact_snapshots(self.ledger,now),0)
        with self.ledger.tx() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM snapshots').fetchone()[0],4)
            self.assertEqual(db.execute('SELECT peak FROM accounts WHERE id=?',('A:crypto',)).fetchone()[0],'21000')
            self.assertEqual(db.execute('SELECT equity FROM snapshots WHERE at LIKE ?',('2026-11-01%',)).fetchone()[0],'20000')
    def test_execution_allowlist(self):
        for sym in ('BTC-EUR','PRIVATE_SYMBOL'):
            self.ledger.execute('A:crypto','SECRET_'+sym,sym,'buy','1',{s:('99','100',self.now) for s in ('BTC-EUR','PRIVATE_SYMBOL')},'.01',market_open=True,approved=True)
        fills=recent_executions(self.ledger)
        self.assertEqual(len(fills),1)
        self.assertEqual(fills[0]['symbol'],'BTC-EUR')
        self.assertNotIn('SECRET',str(fills))
