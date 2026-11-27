import unittest
from ops.stage_equity_panel import probe_args,STATE,CACHE

class PanelStageTests(unittest.TestCase):
    def test_mount_is_read_only_and_isolated(self):
        args=probe_args('sha256:'+'a'*64,'/var/cache/'+CACHE)
        self.assertIn('type=bind,src=/var/cache/'+CACHE+',dst=/equity-report,readonly',args)
        self.assertEqual(args[args.index('--network')+1],'none')
        self.assertIn('--read-only',args)
        self.assertNotIn('/data',str(args))
        self.assertNotIn('/etc/gta-secrets',str(args))
        self.assertNotEqual(STATE,'gta-equity-research')
        self.assertNotEqual(CACHE,'gta-equity-export')
