"""Explicitly approved, schema-compatible updates for the existing VM 106 service.
No scheduler, remote command channel, automatic database restore or volume deletion.
"""
import argparse
import fcntl
import gzip
import json
import os
import re
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import build_opener, ProxyHandler

from ops.stage_release import approved_sha, qualify

ROOT = Path('/opt/global-trading-agent')
BASE = Path('/root/gta-backups')
PROJECT = 'global-trading-agent'
TAG = PROJECT + '-agent:latest'
VOLUME = PROJECT + '_agent-data'
COMPOSE = ['docker', 'compose', '-p', PROJECT, '-f', 'compose.yaml', '-f', 'compose.diagnostics.yaml']


def command(args, *, cwd=ROOT, timeout=120, capture=False, stdout=None):
    result = subprocess.run(args, cwd=cwd, timeout=timeout, check=True,
                            stdout=subprocess.PIPE if capture else stdout, text=capture)
    return result.stdout.strip() if capture else None


def status():
    with build_opener(ProxyHandler({})).open('http://127.0.0.1:8080/api/status', timeout=5) as response:
        return json.load(response)


def image_id(value):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', value):
        raise ValueError('Invalid image ID')
    return value


def check_state(state, version=None, minimum_fills=0, ready=False):
    accounts = state['accounts']
    expected = {p+':'+k for p in 'ABCDE' for k in ('crypto','global','ike')}
    if (state['real_trading'] is not False or state['mode'] != 'crypto-paper'
            or len(accounts) != 15 or {a['account'] for a in accounts} != expected
            or sum(a['fills'] for a in accounts) < minimum_fills):
        raise RuntimeError('Unexpected paper ledger state')
    if version is not None and state['version'] != version:
        raise RuntimeError('Unexpected version')
    if ready and (not state['strategies_running'] or not state['market']['healthy']
                  or state['remote']['state'] != 'ok'):
        raise RuntimeError('Paper market or diagnostics not ready')


def backup_file(path, old, run=command):
    with path.open('xb') as f:
        run(['docker','run','--rm','--network','none','--read-only','--user','0:0',
             '--mount',f'type=volume,src={VOLUME},dst=/data,readonly',
             '--entrypoint','tar',old,'-czf','-','-C','/data','.'], timeout=300, stdout=f)
        f.flush()
        os.fsync(f.fileno())
    # Read full gzip stream for CRC and check that required DBs exist in archive.
    with gzip.open(path,'rb') as f:
        while f.read(1024*1024):
            pass
    with tarfile.open(path,'r:gz') as t:
        names = {m.name.removeprefix('./') for m in t if m.isfile() and m.size>0}
    if not {'ledger.db','market.db'} <= names:
        raise RuntimeError('Backup missing database files')


def current(run=command, read=status):
    config = json.loads(run(COMPOSE+['config','--format','json'],capture=True))
    svc = config['services']['agent']
    env = svc['environment']
    if (config.get('name') != PROJECT or svc.get('image') not in (None,TAG,TAG.removesuffix(':latest'))
            or str(env.get('GTA_PAPER_CRYPTO')) != '1' or env.get('GTA_DIAG_TOKEN_FILE') != '/run/secrets/gta_diagnostics_token'):
        raise RuntimeError('Unexpected Compose configuration')
    cid = run(COMPOSE+['ps','-q','agent'],capture=True)
    if not re.fullmatch(r'[0-9a-f]{64}',cid):
        raise RuntimeError('Expected exactly one running agent')
    obj = json.loads(run(['docker','inspect',cid],capture=True))[0]
    if not obj['State']['Running'] or not any(m.get('Name')==VOLUME and m['Destination']=='/data' for m in obj['Mounts']):
        raise RuntimeError('Unexpected running container or data volume')
    state = read()
    check_state(state,ready=True)
    return image_id(obj['Image']),state


def activate(candidate, version, archive, rollback_tag, old, before, run=command, read=status, save=backup_file, sleep=time.sleep):
    candidate, old = image_id(candidate), image_id(old)
    minimum = sum(a['fills'] for a in before['accounts'])
    if candidate == old:
        return 'already_running'
    run(['docker','tag',old,rollback_tag])
    try:
        run(COMPOSE+['stop','-t','60','agent'],timeout=90)
        save(archive,old,run)
        run(['docker','tag',candidate,TAG])
        run(COMPOSE+['up','-d','--no-build','--pull','never','--force-recreate','--wait','--wait-timeout','90','agent'],timeout=120)
        for attempt in range(30):
            try:
                check_state(read(),version,minimum,ready=True)
                actual,_ = current(run,read)
                if actual != candidate:
                    raise RuntimeError('Candidate image not running')
                return 'activated'
            except Exception:
                if attempt == 29:
                    raise
                sleep(3)
    except Exception:
        # Only schema-compatible releases; never overwrite data or delete volumes.
        run(COMPOSE+['stop','-t','60','agent'],timeout=90)
        run(['docker','tag',old,TAG])
        run(COMPOSE+['up','-d','--no-build','--pull','never','--force-recreate','--wait','--wait-timeout','90','agent'],timeout=120)
        check_state(read(),before['version'],minimum)
        raise RuntimeError('Update failed; previous image restarted. Inspect logs and backup.')


def legacy_compose_guard(root=ROOT, environ=None):
    """The legacy updater must not recreate a service with missing overlays."""
    environ=os.environ if environ is None else environ
    expected='compose.yaml:compose.diagnostics.yaml'
    values=[]
    envfile=Path(root)/'.env'
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            if re.match(r'^\s*(?:export\s+)?COMPOSE_FILE\s*=',line):
                values.append(line.split('=',1)[1].strip())
    if len(values)>1 or any(v!=expected for v in values) or environ.get('COMPOSE_FILE',expected)!=expected:
        raise RuntimeError('Legacy updater cannot preserve additional Compose overlays; use a release-specific migration')
    if any((Path(root)/name).exists() or (Path(root)/name).is_symlink()
           for name in ('compose.equity.yaml','compose.global.yaml')):
        raise RuntimeError('Additional report overlays detected; legacy updater disabled')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approved-sha',type=approved_sha)
    p.add_argument('--expected-version')
    p.add_argument('--apply',action='store_true')
    p.add_argument('--schema-compatible',action='store_true')
    a=p.parse_args()
    if os.geteuid()!=0 or Path.cwd().resolve()!=ROOT:
        p.error('Run as root from /opt/global-trading-agent on VM 106')
    if a.apply and not (a.approved_sha and a.expected_version and a.schema_compatible):
        p.error('Apply requires full approved SHA, version and explicit schema compatibility')
    legacy_compose_guard()
    os.umask(0o077)
    BASE.mkdir(mode=0o700,exist_ok=True)
    with (BASE/'.update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old,before=current()
        if not a.apply:
            print(json.dumps({'ready':True,'running_version':before['version'],'image':old,'fills':sum(x['fills'] for x in before['accounts']),'activated':False},indent=2))
            return
        # Refuse changes to Compose configuration; image-only updates in this tool.
        command(['git','fetch','--depth=1','--no-tags','https://github.com/Siermant/Agent.git',a.approved_sha])
        if command(['git','rev-parse','FETCH_HEAD'],capture=True)!=a.approved_sha:
            raise RuntimeError('Fetched commit differs from approval')
        for name in ('compose.yaml','compose.diagnostics.yaml'):
            if command(['git','show',a.approved_sha+':'+name],capture=True)!=(ROOT/name).read_text().strip():
                raise RuntimeError('Compose changes require manual review')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        manifest=BASE/('qualification-'+stamp+'.json')
        # Git checkout must remain readable by UID 10001 in isolated test containers.
        previous_umask=os.umask(0o022)
        try:
            candidate=qualify(a.approved_sha,manifest)['image_id']
        finally:
            os.umask(previous_umask)
        command(['docker','run','--rm','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','768m','--cpus','1',candidate,'python','-m','global_agent.preflight'],timeout=180)
        # Re-read state after qualification; agent was running throughout.
        latest,before=current()
        if latest!=old:
            raise RuntimeError('Running image changed during qualification')
        archive=BASE/('before-update-'+stamp+'.tar.gz')
        result=activate(candidate,a.expected_version,archive,'gta-rollback:'+stamp,old,before)
        print(json.dumps({'result':result,'commit':a.approved_sha,'image':candidate,'backup':str(archive) if archive.exists() else None},indent=2))
        print('Source files on disk are unchanged; synchronize approved files before any rebuild.')

if __name__=='__main__':
    main()
