import unittest
import tempfile
from pathlib import Path
from datetime import timedelta
from decimal import Decimal as D
from unittest.mock import patch
from global_agent.global_paper_mark import mark
from global_agent.global_paper_initial import open_initial
from global_agent.ledger import Ledger
from test_global_paper_initial import plan,snapshot,NOW,ID

def new_snapshot():
    s=snapshot()
    at=NOW+timedelta(minutes=5)
    s['checked_utc']=at.isoformat()
    s['quotes']={'WMT':dict(s['quotes']['WMT'],bid_pln_reference='420',
                            ask_pln_reference='421',observed_at=at.isoformat())}
    return s

class GlobalMarkTests(unittest.TestCase):
    def test_wait_without_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mark(tmp,lambda s:self.fail('No data request'),lambda:NOW)['state'],'waiting_for_initial_open')
            self.assertFalse((Path(tmp)/'global-paper.db').exists())
    def test_price_mark_and_no_new_fills(self):
        with tempfile.TemporaryDirectory() as tmp:
            open_initial(tmp,plan(),snapshot(),lambda:NOW)
            at=NOW+timedelta(minutes=5)
            def fetch(symbols):
                self.assertEqual(symbols,['WMT'])
                return new_snapshot()
            r=mark(tmp,fetch,lambda:at)
            self.assertEqual(r['state'],'valued')
            self.assertEqual(len(r['accounts']),5)
            self.assertEqual(r['accounts'][0]['invested_pln'],'2100')
            self.assertEqual(r['accounts'][0]['fills'],1)
            self.assertEqual(r['accounts'][-1]['equity_pln'],'50000')
            self.assertEqual(len(r['history']),1)
            self.assertEqual(D(r['history'][0]['equity_pln']['A:global']),D(r['accounts'][0]['equity_pln']))
            self.assertFalse(r['corporate_actions_verified'])
    def test_closed_or_stale_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            open_initial(tmp,plan(),snapshot(),lambda:NOW)
            ledger=Ledger(Path(tmp)/'global-paper.db')
            before=ledger.status()
            self.assertEqual(mark(tmp,lambda _: {'ready':False,'state':'market_closed'},lambda:NOW)['state'],'market_closed')
            with self.assertRaises(ValueError):mark(tmp,lambda _:new_snapshot(),lambda:NOW+timedelta(minutes=10))
            self.assertEqual(ledger.status(),before)
    def test_atomic_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            open_initial(tmp,plan(),snapshot(),lambda:NOW)
            ledger=Ledger(Path(tmp)/'global-paper.db')
            before=ledger.status()
            original=Ledger._snapshot
            def fail(self,db,row,equity,now):
                if row['id']=='B:global':raise RuntimeError('fixture')
                return original(self,db,row,equity,now)
            with patch.object(Ledger,'_snapshot',fail):
                with self.assertRaises(RuntimeError):mark(tmp,lambda _:new_snapshot(),lambda:NOW+timedelta(minutes=5))
            self.assertEqual(ledger.status(),before)
