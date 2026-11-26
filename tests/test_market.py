import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from global_agent.market import parse, QuoteStore
from global_agent.__main__ import view, render
from global_agent.ledger import Ledger

class MarketTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 26, 12, tzinfo=timezone.utc)
        self.item = {"instId":"BTC-EUR","instType":"SPOT","bidPx":"99","askPx":"100","last":"99.5","ts":str(int(self.now.timestamp()*1000))}
    def test_valid(self):
        rows = parse({"code":"0","data":[self.item]}, self.now)
        self.assertEqual(rows[0][2:], ("99","100","99.5"))
    def test_reject_bad(self):
        for change in ({"bidPx":"NaN"},{"bidPx":"101"},{"instId":"BAD"},{"instType":"SWAP"},{"ts":"0"},{"ts":str(int((self.now+timedelta(seconds=1)).timestamp()*1000))}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                parse({"code":"0","data":[dict(self.item,**change)]},self.now)
    def test_feed_failure_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=QuoteStore(Path(tmp)/"market.db")
            self.assertFalse(s.status(self.now)["healthy"])
            rows=parse({"code":"0","data":[self.item]},self.now)
            s.record(rows,self.now)
            self.assertTrue(s.status(self.now)["healthy"])
            self.assertFalse(s.status(self.now+timedelta(seconds=61))["healthy"])
            s.fail(self.now,"test")
            self.assertFalse(s.status(self.now)["quotes"][0]["fresh"])
    def test_dashboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=view(Ledger(Path(tmp)/"ledger.db"),QuoteStore(Path(tmp)/"market.db"))
            self.assertEqual(len(state["accounts"]),15)
            self.assertFalse(state["real_trading"])
            self.assertIn("handel zablokowany",render(state))
            self.assertIn("E:ike",render(state))

if __name__=="__main__":
    unittest.main()
