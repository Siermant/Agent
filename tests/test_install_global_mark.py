import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
from ops import install_global_mark as m
from ops.update_release import legacy_compose_guard

class MarkInstallTests(unittest.TestCase):
    def test_no_opening_no_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m,'LEDGER',Path(tmp)/'missing.db'),patch.object(m,'command') as cmd:
                self.assertEqual(m.install('not-read')['state'],'waiting_for_initial_open')
                cmd.assert_not_called()
    def test_install_and_failed_first_mark(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                dest=root/'mark';dest.mkdir();(dest/'old').write_text('old')
                units=root/'units';units.mkdir()
                source=root/'stage';source.mkdir()
                for n in ('global_agent','tests','ops'):(source/n).mkdir()
                for n in (m.SERVICE,m.TIMER):(source/'ops'/n).write_text('unit')
                ledger=root/'ledger.db';ledger.touch()
                image='sha256:'+'a'*64
                calls=[]
                def cmd(args,**kw):
                    calls.append(args)
                    if args[:3]==['docker','image','inspect']:return image
                    if args[:2]==['systemctl','show']:return 'active' if args[2]==m.INITIAL else 'inactive'
                    return ''
                def tempdir(**kw):
                    p=root/'temp';p.mkdir();return str(p)
                report={'valued_at':'2026-11-27T15:00:00+00:00','accounts':[{'fills':1}]}
                with ExitStack() as st:
                    for k,v in {'DEST':dest,'UNIT_ROOT':units,'SOURCE':source,'LEDGER':ledger}.items():st.enter_context(patch.object(m,k,v))
                    st.enter_context(patch.object(m,'qualification',return_value=image))
                    st.enter_context(patch.object(m,'running'))
                    st.enter_context(patch.object(m,'command',side_effect=cmd))
                    st.enter_context(patch.object(m,'actual_reports'))
                    st.enter_context(patch.object(m.tempfile,'mkdtemp',side_effect=tempdir))
                    st.enter_context(patch.object(m,'check_report',side_effect=RuntimeError('no mark') if fail else None,return_value=report))
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'no mark'):m.install('q')
                        self.assertTrue((dest/'old').exists())
                        self.assertFalse((units/m.TIMER).exists())
                        self.assertNotIn(['systemctl','stop',m.INITIAL],calls)
                    else:
                        self.assertEqual(m.install('q')['state'],'valuation_active')
                        self.assertTrue((units/m.TIMER).exists())
                        self.assertIn(['systemctl','enable','--now',m.TIMER],calls)
                        self.assertIn(['systemctl','stop',m.INITIAL],calls)
                    self.assertTrue(ledger.exists())
    def test_legacy_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            legacy_compose_guard(root,{})
            (root/'.env').write_text('COMPOSE_FILE=compose.yaml:compose.diagnostics.yaml:compose.equity.yaml\n')
            with self.assertRaises(RuntimeError):legacy_compose_guard(root,{})
            (root/'.env').write_text('COMPOSE_FILE=compose.yaml:compose.diagnostics.yaml\n')
            with self.assertRaises(RuntimeError):legacy_compose_guard(root,{'COMPOSE_FILE':'other.yaml'})
            (root/'compose.global.yaml').touch()
            with self.assertRaises(RuntimeError):legacy_compose_guard(root,{})
