"""Offline HTTP acceptance tests using the real handler and temporary databases."""
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler
from zipfile import ZipFile
from global_agent.__main__ import handler, VERSION
from global_agent.ledger import Ledger
from global_agent.market import QuoteStore


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        now = datetime.now(timezone.utc)
        cls.ledger = Ledger(root/'ledger.db', clock=lambda: now)
        cls.store = QuoteStore(root/'market.db')
        cls.ledger.execute('A:crypto','LOCAL_TEST_ORDER','BTC-EUR','buy','1',
                           {'BTC-EUR':('99','100',now)},'.01',
                           market_open=True,approved=True)
        cls.server = ThreadingHTTPServer(('127.0.0.1',0), handler(cls.ledger,cls.store))
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'
        cls.opener = build_opener(ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def get(self,path):
        return self.opener.open(self.base+path,timeout=5)

    def test_health(self):
        with self.get('/healthz') as res:
            self.assertEqual(res.read(),b'ok')

    def test_status(self):
        with self.get('/api/status') as res:
            s=json.load(res)
        self.assertEqual(s['version'],VERSION)
        self.assertEqual(len(s['accounts']),15)
        self.assertEqual(s['mode'],'monitor-only')
        self.assertFalse(s['real_trading'])
        self.assertFalse(s['strategies_running'])
        self.assertEqual(s['recent_executions'][0]['symbol'],'BTC-EUR')
        self.assertNotIn('LOCAL_TEST_ORDER',str(s))

    def test_dashboard(self):
        with self.get('/') as res:
            page=res.read().decode()
            self.assertEqual(res.headers['Cache-Control'],'no-store')
        self.assertIn('Ostatnie transakcje symulacyjne',page)
        self.assertIn('Pobierz raport.zip',page)
        self.assertIn('handel zablokowany',page)

    def test_report(self):
        with self.get('/raport.zip') as res:
            self.assertIn('raport.zip',res.headers['Content-Disposition'])
            with ZipFile(BytesIO(res.read())) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(len(json.loads(z.read('executions.json'))),1)
                self.assertNotIn(b'LOCAL_TEST_ORDER',z.read('executions.json'))

    def test_no_remote_commands(self):
        with self.assertRaises(HTTPError) as err:
            self.opener.open(Request(self.base+'/api/status',data=b'{}',method='POST'),timeout=5)
        self.assertEqual(err.exception.code,501)
        with self.assertRaises(HTTPError) as err:
            self.get('/api/execute')
        self.assertEqual(err.exception.code,404)
        self.assertEqual(sum(a['fills'] for a in self.ledger.status()),1)
