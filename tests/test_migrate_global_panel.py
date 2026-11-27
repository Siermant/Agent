import json
import unittest
import tempfile
from pathlib import Path
from datetime import datetime,timedelta,timezone
from ops import migrate_global_panel as m
from ops.stage_global_panel import fixture,CANDIDATE
from global_agent.global_panel import summary

NOW=datetime(2026,11,27,15,tzinfo=timezone.utc)
IMAGE='sha256:'+'a'*64

class GlobalMigrationTests(unittest.TestCase):
    def test_env_only_adds_global_overlay(self):
        original='GTA_PAPER_CRYPTO=1\nCOMPOSE_FILE='+':'.join(m.FILES)+'\nGTA_DIAG_TOKEN_PATH=/etc/gta-secrets/github-token\n'
        self.assertEqual(m.next_env(original),original.replace(':'.join(m.FILES),':'.join((*m.FILES,'compose.global.yaml'))))
        for x in ('',original+original,original.replace('compose.equity.yaml','other.yaml')):
            with self.assertRaises(ValueError):m.next_env(x)
    def test_qualification_gate(self):
        v={'commit':CANDIDATE,'activated':False,'checked_utc':NOW.isoformat(),'image_id':IMAGE,
           **{k:'passed' for k in ('unit_tests','offline_http_smoke','global_fixture_http','existing_equity_mount')}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'q.json';p.write_text(json.dumps(v))
            self.assertEqual(m.qualification(p,NOW),IMAGE)
            with self.assertRaises(ValueError):m.qualification(p,NOW+timedelta(days=2))
            for change in ({'commit':'b'*40},{'global_fixture_http':'failed'},{'activated':True}):
                p.write_text(json.dumps(dict(v,**change)))
                with self.assertRaises(ValueError):m.qualification(p,NOW)
    def test_synthetic_fixture_not_live_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            s=summary(tmp)
            self.assertEqual(s['state'],'valued')
            self.assertEqual(sum(a['fills'] for a in s['accounts']),0)
            state={'real_trading':False,'version':'0.15.0-dev','mode':'crypto-paper',
                   'accounts':[{'account':p+':'+k,'fills':0} for p in 'ABCDE' for k in ('crypto','global','ike')],
                   'market':{'healthy':True},'remote':{'state':'ok'},'strategies_running':True,
                   'equity_research':{'state':'ok','instruments':28},'global_paper':s}
            with self.assertRaisesRegex(RuntimeError,'Global valuation'):m.panel_ok(state,0,'0.15.0-dev')
    def test_rollback_and_success(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);base=root/'backups';base.mkdir()
                original='COMPOSE_FILE='+':'.join(m.FILES)+'\n'
                env=root/'.env';env.write_text(original)
                for f in m.FILES:(root/f).write_text('old config')
                old='sha256:'+'b'*64
                before={'accounts':[{'fills':7}]}
                calls=[]
                def cmd(args,**kw):
                    calls.append(args)
                    if args[:3]==['docker','image','inspect']:return IMAGE
                    if args[:2]==['systemctl','is-enabled']:return 'enabled'
                    if args[:2]==['git','rev-parse']:return CANDIDATE
                    if args[:2]==['git','show']:return 'new config' if args[-1].endswith('compose.global.yaml') else 'old config'
                    if args==m.NEW+m.UP and fail:raise RuntimeError('simulated activation failure')
                    return ''
                def running(compose,version,minimum=0):
                    return (old,before) if compose==m.OLD else (IMAGE,before)
                with ExitStack() as st:
                    for k,v in {'ROOT':root,'BASE':base}.items():st.enter_context(patch.object(m,k,v))
                    st.enter_context(patch.object(m,'qualification',return_value=IMAGE))
                    st.enter_context(patch.object(m,'command',side_effect=cmd))
                    st.enter_context(patch.object(m,'running',side_effect=running))
                    st.enter_context(patch.object(m,'actual_reports'))
                    st.enter_context(patch.object(m,'backup_file'))
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'previous configuration restored'):m.migrate(root/'q')
                        self.assertEqual(env.read_text(),original)
                        self.assertFalse((root/'compose.global.yaml').exists())
                        self.assertIn(m.OLD+m.UP,calls)
                    else:
                        self.assertTrue(m.migrate(root/'q')['activated'])
                        self.assertEqual(env.read_text(),m.next_env(original))
                        self.assertTrue((root/'compose.global.yaml').exists())
                    self.assertIn(m.OLD+['stop','-t','60','agent'],calls)
