"""Qualify an explicitly approved commit; never modify the running service."""
import argparse
import json
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = 'https://github.com/Siermant/Agent.git'
FILES = ('global_agent', 'tests', 'Dockerfile', '.dockerignore', 'ops')
SAFE = ['--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--memory', '768m', '--cpus', '1',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m']


def command(args, cwd=None, timeout=120, capture=False):
    return subprocess.run(args, cwd=cwd, timeout=timeout, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None).stdout


def approved_sha(value):
    if not re.fullmatch(r'[0-9a-f]{40}', value):
        raise ValueError('Full lowercase approved commit SHA required')
    return value


def qualify(sha, output, run=command, sleep=time.sleep):
    sha = approved_sha(sha)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new qualification manifest path')
    tag = 'gta-candidate:' + sha
    with tempfile.TemporaryDirectory(prefix='gta-stage-') as tmp:
        root = Path(tmp)
        run(['git', 'init', '--quiet'], cwd=root)
        run(['git', 'fetch', '--no-tags', '--depth=1', REPO, sha], cwd=root)
        actual = run(['git', 'rev-parse', 'FETCH_HEAD'], cwd=root, capture=True).strip()
        if actual != sha:
            raise RuntimeError('Fetched commit differs from approval')
        # Extract only the required paths, never legacy source/logs or credentials.
        run(['git', 'checkout', sha, '--', *FILES], cwd=root)
        run(['docker', 'build', '--tag', tag, str(root)], timeout=900)
        image = run(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag], capture=True).strip()
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise RuntimeError('Invalid built image ID')
        run(['docker', 'run', '--rm', *SAFE, '--mount',
             f'type=bind,src={root / "tests"},dst=/app/tests,readonly', '--mount',
             f'type=bind,src={root / "ops"},dst=/app/ops,readonly',
             image, 'python', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], timeout=300)
        cid = run(['docker', 'run', '-d', *SAFE, '--tmpfs',
                   '/data:rw,nosuid,nodev,size=128m,uid=10001,gid=10001,mode=0700',
                   '-e', 'GTA_PAPER_CRYPTO=0', image], capture=True).strip()
        if not re.fullmatch(r'[0-9a-f]{64}', cid):
            raise RuntimeError('Invalid candidate container ID')
        try:
            probe = ('import json,urllib.request as u; '
                     's=json.load(u.urlopen("http://127.0.0.1:8080/api/status",timeout=3)); '
                     'assert s["real_trading"] is False and s["mode"]=="monitor-only"; '
                     'assert len(s["accounts"])==15; '
                     'assert u.urlopen("http://127.0.0.1:8080/healthz",timeout=3).read()==b"ok"')
            for attempt in range(15):
                try:
                    run(['docker', 'exec', cid, 'python', '-c', probe], timeout=10)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if attempt == 14:
                        raise RuntimeError('Candidate HTTP smoke test failed')
                    sleep(2)
        finally:
            run(['docker', 'rm', '-f', cid], timeout=30)
    result = {'commit':sha, 'image_id':image, 'tag':tag,
              'qualified_utc':datetime.now(timezone.utc).isoformat(),
              'unit_tests':'passed', 'offline_http_smoke':'passed',
              'live_market_test':'not_run', 'activated':False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=output.parent, delete=False) as f:
        json.dump(result, f, indent=2)
        temp = Path(f.name)
    temp.replace(output)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approved-sha', required=True, type=approved_sha)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(qualify(a.approved_sha, a.output), indent=2))


if __name__ == '__main__':
    main()
