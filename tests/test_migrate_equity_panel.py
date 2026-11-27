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
