import json
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from ops.migrate_equity_panel import qualification,env_update,CANDIDATE,OLD_FILES,NEW_FILES

NOW=datetime(2026,11,27,12,tzinfo=timezone.utc)
IMAGE='sha256:'+'a'*64

class PanelMigrationTests(unittest.TestCase):
    def test_env_only_changes_compose_file(self):
        orig='GTA_PAPER_CRYPTO=1\nCOMPOSE_FILE='+OLD_FILES+'\nGTA_DIAG_TOKEN_PATH=/etc/gta-secrets/github-token\n'
        self.assertEqual(env_update(orig),orig.replace(OLD_FILES,NEW_FILES))
        for invalid in ('', 'COMPOSE_FILE=unexpected\n',orig+'COMPOSE_FILE='+OLD_FILES):
            with self.assertRaises(ValueError):
                env_update(invalid)
    def test_qualification_gate(self):
        m={'commit':CANDIDATE,'image_id':IMAGE,'checked_utc':NOW.isoformat(),
           'activated':False,'instruments':28,'cohorts':2,
           'offline_qualification':'passed','isolated_provider_scan':'passed','container_export_read':'passed'}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'
            p.write_text(json.dumps(m))
            self.assertEqual(qualification(p,NOW),IMAGE)
            with self.assertRaises(ValueError):
                qualification(p,NOW+timedelta(days=2))
            for change in ({'commit':'b'*40},{'container_export_read':'failed'},{'instruments':4},{'activated':True}):
                p.write_text(json.dumps(dict(m,**change)))
                with self.assertRaises(ValueError):
                    qualification(p,NOW)


    def test_migration_and_rollback(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        from ops import migrate_equity_panel as m
        for fail in (False,True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                research=root/'research';research.mkdir()
                (research/'old').write_text('old')
                base=root/'backups';base.mkdir()
                stage=root/'staged';stage.mkdir()
                unit=root/'gta-equity-research.service';unit.write_text('old unit')
                env=root/'.env';original='COMPOSE_FILE='+OLD_FILES+'\n';env.write_text(original)
                overlay=root/'compose.equity.yaml'
                old='sha256:'+'b'*64
                before={'version':'0.13.0-dev','crypto_universe':['BTC-EUR','ETH-EUR'],'accounts':[{'fills':7}]}
                calls=[]
                def cmd(args,**kwargs):
                    calls.append(args)
                    if args[:3]==['docker','image','inspect']: return IMAGE
                    if args[:2]==['systemctl','is-enabled']: return 'enabled'
                    if 'config' in args:
                        return json.dumps({'services':{'agent':{'environment':{'GTA_EQUITY_REPORT_DIR':'/equity-report'},'volumes':[{'source':str(root/'export'),'target':'/equity-report','read_only':True}]}}})
                    if 'ps' in args:return 'c'*64
                    if args[:2]==['docker','inspect']:
                        return json.dumps([{'Image':IMAGE,'Mounts':[{'Destination':'/equity-report','RW':False}]}])
                    return ''
                def prep(sha,path):
                    (path/'ops').mkdir()
                    (path/'ops'/unit.name).write_text('new unit')
                    (path/'compose.equity.yaml').write_text('overlay')
                    (path/'new').write_text('new')
                with ExitStack() as st:
                    for key,value in {'ROOT':root,'BASE':base,'RESEARCH':research,'UNIT':unit,'EXPORT':root/'export'}.items():
                        st.enter_context(patch.object(m,key,value))
                    # Avoid touching actual /opt while retaining real directory renames.
                    # The implementation stores the rollback path, so replace Path only for its /opt backup.
                    real_path=Path
                    def local_path(value):
                        return root/real_path(value).name if str(value).startswith('/opt/gta-equity-pre-panel-') else real_path(value)
                    st.enter_context(patch.object(m,'Path',side_effect=local_path))
                    st.enter_context(patch.object(m.tempfile,'mkdtemp',return_value=str(stage)))
                    st.enter_context(patch.object(m,'qualification',return_value=IMAGE))
                    st.enter_context(patch.object(m,'current',return_value=(old,before)))
                    st.enter_context(patch.object(m,'command',side_effect=cmd))
                    st.enter_context(patch.object(m,'prepare',side_effect=prep))
                    st.enter_context(patch.object(m,'backup_file'))
                    st.enter_context(patch.object(m,'wait_panel',side_effect=RuntimeError('test') if fail else None))
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'previous configuration restarted'):
                            m.migrate(root/'manifest')
                        self.assertEqual(env.read_text(),original)
                        self.assertEqual(unit.read_text(),'old unit')
                        self.assertTrue((research/'old').exists())
                        self.assertFalse(overlay.exists())
                        self.assertIn(['docker','tag',old,m.TAG],calls)
                    else:
                        self.assertTrue(m.migrate(root/'manifest')['activated'])
                        self.assertEqual(env.read_text(),env_update(original))
                        self.assertEqual(unit.read_text(),'new unit')
                        self.assertTrue((research/'new').exists())
                        self.assertTrue(overlay.exists())
                self.assertIn(['systemctl','start',m.TIMER],calls)
