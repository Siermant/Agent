import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from global_agent.remote_diag import GitHubSink, Outbox, snapshots, run_cycle, TARGET

NOW=datetime(2026,11,26,12,tzinfo=timezone.utc)
STATE={'version':'test','mode':'monitor-only','paper':{'enabled':False,'state':'disabled'},'market':{'healthy':False,'checked':None,'secret':'SECRET'},'statistics':[],'secret':'SECRET'}

class MockSink:
    def __init__(self): self.sent=[]; self.fail=False
    def send(self,k,p):
        if self.fail: raise OSError('offline')
        self.sent.append((k,p))

class RemoteTests(unittest.TestCase):
    def test_public_blocked_before_write(self):
        sink=GitHubSink('test')
        calls=[]
        def request(method,path,payload=None):
            calls.append(method)
            return {'full_name':TARGET,'private':False,'visibility':'public','fork':False}
        sink.request=request
        with self.assertRaises(PermissionError): sink.send('status',{})
        self.assertEqual(calls,['GET'])
    def test_private_update(self):
        sink=GitHubSink('test'); calls=[]
        def request(method,path,payload=None):
            calls.append((method,path,payload))
            if path=='': return {'full_name':TARGET,'private':True,'visibility':'private','fork':False,'default_branch':'main'}
            if method=='GET': return {'type':'file','sha':'a'*40}
            return {}
        sink.request=request
        sink.send('status',{'real_trading':False})
        self.assertEqual(calls[-1][0:2],('PUT','/contents/status.json'))
        self.assertEqual(calls[-1][2]['sha'],'a'*40)
    def test_restart_and_outage(self):
        with tempfile.TemporaryDirectory() as t:
            sink=MockSink(); sink.fail=True
            with self.assertRaises(OSError): run_cycle(Outbox(t),sink,STATE,[],NOW)
            self.assertEqual(len(list(Path(t).glob('*.json'))),3)
            sink.fail=False
            run_cycle(Outbox(t),sink,STATE,[],NOW+timedelta(minutes=16))
            self.assertEqual(len(sink.sent),3)
            run_cycle(Outbox(t),sink,STATE,[],NOW+timedelta(minutes=17))
            self.assertEqual(len(sink.sent),3)
            run_cycle(Outbox(t),sink,STATE,[],NOW+timedelta(hours=1))
            self.assertEqual(len(sink.sent),5)
    def test_health_change_and_allowlist(self):
        a=snapshots(STATE,[],NOW)
        b=snapshots(dict(STATE,market={'healthy':True,'checked':None}),[],NOW)
        self.assertNotEqual(a['status'][0],b['status'][0])
        self.assertNotIn('SECRET',json.dumps(a))
        self.assertFalse(a['status'][1]['real_trading'])
