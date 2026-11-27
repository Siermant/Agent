import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from decimal import Decimal as D
from pathlib import Path
from global_agent.preflight import check, resources
from global_agent.paper_live import UNIVERSE, ALL_UNIVERSE

NOW = datetime(2026,11,26,12,tzinfo=timezone.utc)

def context(day):
    return day-timedelta(days=1), D(4), {s:[D(100)]*20 for s in UNIVERSE}

class PreflightTests(unittest.TestCase):
    def test_ready(self):
        r=check(lambda:[(s,) for s in UNIVERSE],context,lambda:NOW)
        self.assertTrue(r['ready'])
        self.assertFalse(r['real_trading'])
    def test_incomplete(self):
        r=check(lambda:[('BTC-EUR',)],context,lambda:NOW)
        self.assertFalse(r['ready'])
    def test_error_redacted(self):
        def fail(day): raise OSError('SECRET')
        r=check(lambda:[],fail,lambda:NOW)
        self.assertFalse(r['ready'])
        self.assertNotIn('SECRET',str(r))
    def test_resources(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            (p/'memory.current').write_text('123')
            (p/'memory.max').write_text('max')
            (p/'cpu.stat').write_text('usage_usec 456\nuser_usec 123\n')
            r=resources(p,p)
            self.assertEqual(r['memory_bytes'],123)
            self.assertIsNone(r['memory_limit_bytes'])
            self.assertEqual(r['cpu_usage_usec_total'],456)
            self.assertGreater(r['disk_total_bytes'],0)

    def test_all_universe(self):
        all_context=lambda day:(day-timedelta(days=1),D(4),{s:[D(100)]*20 for s in ALL_UNIVERSE})
        r=check(lambda:[(s,) for s in ALL_UNIVERSE],all_context,lambda:NOW,ALL_UNIVERSE)
        self.assertTrue(r['ready'])
        r=check(lambda:[(s,) for s in UNIVERSE],all_context,lambda:NOW,ALL_UNIVERSE)
        self.assertFalse(r['ready'])
        r=check(lambda:[(s,) for s in ALL_UNIVERSE],context,lambda:NOW,ALL_UNIVERSE)
        self.assertFalse(r['ready'])
