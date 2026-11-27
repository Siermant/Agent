import tempfile
import unittest
import sqlite3
from contextlib import closing
from pathlib import Path
from datetime import date,timedelta
from global_agent.corporate_actions import scan
from global_agent.corporate_audit import record
from global_agent.equity_quotes import SYMBOLS
from test_corporate_actions import event,payload
from test_global_paper_initial import NOW

START=date(2026,11,1)
END=date(2026,12,20)
def report(events):
    return scan(lambda _:payload({'cash_dividends':events}),SYMBOLS,START,END)

class CorporateAuditTests(unittest.TestCase):
    def test_revisions_restart_and_missing(self):
        e=event()
        with tempfile.TemporaryDirectory() as tmp:
            r=record(tmp,report([e]),NOW)
            self.assertEqual((r['new_events'],r['total_revisions']),(1,1))
            self.assertEqual(record(tmp,report([e]),NOW)['state'],'already_recorded')
            r=record(tmp,report([e]),NOW+timedelta(days=1))
            self.assertEqual((r['new_events'],r['changed_events'],r['total_revisions']),(0,0,1))
            changed=dict(e,rate='0.50')
            r=record(tmp,report([changed]),NOW+timedelta(days=2))
            self.assertEqual((r['changed_events'],r['total_revisions']),(1,2))
            r=record(tmp,report([]),NOW+timedelta(days=3))
            self.assertEqual(r['missing_in_window'],1)
            self.assertTrue(r['requires_review'])
            r=record(tmp,report([changed]),NOW+timedelta(days=4))
            self.assertEqual((r['reappeared_events'],r['total_revisions']),(1,2))
            with closing(sqlite3.connect(Path(tmp)/'corporate-audit.db')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM scans').fetchone()[0],5)
                self.assertEqual(db.execute('SELECT count(*) FROM revisions').fetchone()[0],2)
    def test_conflicting_same_timestamp_and_clock_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            e=event()
            record(tmp,report([e]),NOW)
            with self.assertRaises(ValueError):record(tmp,report([dict(e,rate='1')]),NOW)
            with self.assertRaises(ValueError):record(tmp,report([e]),NOW-timedelta(minutes=1))
            with closing(sqlite3.connect(Path(tmp)/'corporate-audit.db')) as db:
                self.assertEqual(db.execute('SELECT count(*) FROM scans').fetchone()[0],1)
    def test_overlap_and_no_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            v=report([event(),event()])
            v['events'][0]['secret']='not_for_database'
            r=record(tmp,v,NOW)
            self.assertEqual(len(r['overlapping_cash_events']),1)
            self.assertTrue(r['requires_review'])
            self.assertFalse(r['accounting_enabled'])
            with closing(sqlite3.connect(Path(tmp)/'corporate-audit.db')) as db:
                self.assertFalse(any('not_for_database' in row[0] for row in db.execute('SELECT payload FROM revisions')))
    def test_reject_partial_universe_and_wrong_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            v=report([event()])
            for invalid in (dict(v,symbols_checked=1),dict(v,accounting_enabled=True)):
                with self.assertRaises(ValueError):record(tmp,invalid,NOW)
            self.assertFalse((Path(tmp)/'corporate-audit.db').exists())
