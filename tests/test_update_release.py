import tempfile
import unittest
from pathlib import Path
from ops.update_release import activate, check_state, TAG, COMPOSE
from unittest.mock import patch

OLD='sha256:'+'a'*64
NEW='sha256:'+'b'*64

def state(version='0.11',fills=7,ready=True):
    return {'version':version,'real_trading':False,'mode':'crypto-paper',
            'accounts':[{'account':p+':'+k,'fills':fills if p+k=='Acrypto' else 0} for p in 'ABCDE' for k in ('crypto','global','ike')],
            'strategies_running':ready,'market':{'healthy':ready},'remote':{'state':'ok' if ready else 'waiting'}}

class UpdateTests(unittest.TestCase):
    def test_ledger_guard(self):
        with self.assertRaises(RuntimeError): check_state(state(fills=6),minimum_fills=7)
        with self.assertRaises(RuntimeError): check_state(state(ready=False),ready=True)
        s=state(); s['real_trading']=True
        with self.assertRaises(RuntimeError): check_state(s)
    def test_same_image(self):
        self.assertEqual(activate(OLD,'0.11',Path('/unused'),'rollback',OLD,state()),'already_running')
    def test_success(self):
        calls=[]
        def run(args,**kw): calls.append(args)
        def save(path,old,run): calls.append(['backup'])
        with patch('ops.update_release.current',return_value=(NEW,state('0.12'))):
            result=activate(NEW,'0.12',Path('/unused'),'rollback',OLD,state(),run,lambda:state('0.12'),save,lambda _:None)
        self.assertEqual(result,'activated')
        self.assertLess(calls.index(['backup']),calls.index(['docker','tag',NEW,TAG]))
    def test_backup_failure_restores(self):
        calls=[]
        def run(args,**kw): calls.append(args)
        def save(*args): raise OSError('disk full')
        with self.assertRaisesRegex(RuntimeError,'previous image restarted'):
            activate(NEW,'0.12',Path('/unused'),'rollback',OLD,state(),run,state,save,lambda _:None)
        self.assertNotIn(['docker','tag',NEW,TAG],calls)
        self.assertIn(['docker','tag',OLD,TAG],calls)
    def test_unhealthy_candidate_rolls_back(self):
        calls=[]
        active=[OLD]
        def run(args,**kw):
            calls.append(args)
            if args[:2]==['docker','tag'] and args[-1]==TAG: active[0]=args[2]
        def read(): return state('0.12',ready=False) if active[0]==NEW else state()
        with self.assertRaisesRegex(RuntimeError,'previous image restarted'):
            activate(NEW,'0.12',Path('/unused'),'rollback',OLD,state(),run,read,lambda *a:None,lambda _:None)
        self.assertEqual(active[0],OLD)
        self.assertFalse(any('down' in c or 'volume' in c for c in calls))
