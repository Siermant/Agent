import subprocess
import tempfile
import unittest
from pathlib import Path
from ops.stage_release import qualify, approved_sha

SHA = 'a'*40
IMAGE = 'sha256:'+'b'*64
CID = 'c'*64


class StageTests(unittest.TestCase):
    def fake(self, fail=None):
        calls = []
        def run(args, **kwargs):
            calls.append(args)
            if fail and fail(args):
                raise subprocess.CalledProcessError(1, args)
            if args[:2] == ['git','rev-parse']:
                return SHA+'\n'
            if args[:3] == ['docker','image','inspect']:
                return IMAGE+'\n'
            if args[:3] == ['docker','run','-d']:
                return CID+'\n'
        return run, calls

    def test_sha(self):
        for s in ('develop','main','abc','a'*39,'A'*40):
            with self.assertRaises(ValueError):
                approved_sha(s)

    def test_success_no_live_service(self):
        with tempfile.TemporaryDirectory() as t:
            run,calls=self.fake()
            p=Path(t)/'qualified.json'
            r=qualify(SHA,p,run=run,sleep=lambda _:None)
            self.assertFalse(r['activated'])
            self.assertEqual(r['image_id'],IMAGE)
            self.assertTrue(p.exists())
            self.assertTrue(all('--network' in x and x[x.index('--network')+1]=='none' for x in calls if x[:2]==['docker','run']))
            self.assertFalse(any(x[:2]==['docker','compose'] for x in calls))
            self.assertIn(['docker','rm','-f',CID],calls)

    def test_failure_removes_candidate(self):
        with tempfile.TemporaryDirectory() as t:
            run,calls=self.fake(lambda a:a[:2]==['docker','exec'])
            p=Path(t)/'qualified.json'
            with self.assertRaises(RuntimeError):
                qualify(SHA,p,run=run,sleep=lambda _:None)
            self.assertFalse(p.exists())
            self.assertIn(['docker','rm','-f',CID],calls)

    def test_failed_unit_tests_do_not_start(self):
        with tempfile.TemporaryDirectory() as t:
            run,calls=self.fake(lambda a:a[:3]==['docker','run','--rm'])
            with self.assertRaises(subprocess.CalledProcessError):
                qualify(SHA,Path(t)/'qualified.json',run=run)
            self.assertFalse(any(x[:3]==['docker','run','-d'] for x in calls))
