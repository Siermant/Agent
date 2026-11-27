"""Explicit v0.14 migration. Requires successful same-commit VM qualification.
No live orders, volume replacement, automatic DB restore or credential export.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from ops.stage_release import approved_sha, REPO
from ops.stage_equity_panel import probe_args
from ops.update_release import ROOT, BASE, COMPOSE, TAG, current, check_state, status, image_id, backup_file, command

RESEARCH=Path('/opt/gta-equity-research')
UNIT=Path('/etc/systemd/system/gta-equity-research.service')
TIMER='gta-equity-research.timer'
SERVICE='gta-equity-research.service'
EXPORT=Path('/var/cache/gta-equity-export')
OLD_FILES='compose.yaml:compose.diagnostics.yaml'
NEW_FILES=OLD_FILES+':compose.equity.yaml'
CANDIDATE='74b1c5638414e46eaa599cf01a1568773ffc1204'


def qualification(manifest, now=None):
    m=json.loads(Path(manifest).read_text())
    now=now or datetime.now(timezone.utc)
    checked=datetime.fromisoformat(m['checked_utc'])
    if (m['commit']!=CANDIDATE or m.get('activated') is not False
            or m.get('instruments')!=28 or m.get('cohorts')!=2
            or any(m.get(k)!='passed' for k in ('offline_qualification','isolated_provider_scan','container_export_read'))
            or not timedelta(0)<=now-checked<=timedelta(days=1)):
        raise ValueError('Fresh successful v0.14 qualification required')
    return image_id(m['image_id'])


def env_update(original):
    lines=original.splitlines(keepends=True)
    hits=[i for i,l in enumerate(lines) if l.strip().startswith('COMPOSE_FILE=')]
    if len(hits)!=1 or lines[hits[0]].strip()!='COMPOSE_FILE='+OLD_FILES:
        raise ValueError('Unexpected COMPOSE_FILE; manual review required')
    i=hits[0]
    lines[i]='COMPOSE_FILE='+NEW_FILES+'\n'
    return ''.join(lines)


def replace_text(path,text,mode):
    temp=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
            temp=Path(f.name)
            f.write(text)
            f.flush()
            os.fchmod(f.fileno(),mode)
            os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def panel_ok(state,minimum):
    check_state(state,'0.14.0-dev',minimum,ready=True)
    e=state['equity_research']
    if e.get('state')!='ok' or e.get('instruments')!=28 or len(e.get('cohorts',[]))!=2:
        raise RuntimeError('Dashboard ranking unavailable')


def wait_panel(minimum, read=status, sleep=time.sleep):
    for i in range(30):
        try:
            panel_ok(read(),minimum)
            return
        except Exception:
            if i==29:
                raise
            sleep(3)


def prepare(sha,root):
    command(['git','fetch','--depth=1','--no-tags',REPO,sha])
    if command(['git','rev-parse','FETCH_HEAD'],capture=True)!=sha:
        raise RuntimeError('Commit mismatch')
    for name in ('compose.yaml','compose.diagnostics.yaml'):
        if command(['git','show',sha+':'+name],capture=True)!=(ROOT/name).read_text().strip():
            raise RuntimeError('Base Compose differs')
    # Extract only reviewed paths; do not copy the checkout's secrets or legacy files.
    archive=root/'source.tar'
    with archive.open('wb') as f:
        command(['git','archive',sha,'global_agent','tests','ops','compose.equity.yaml'],stdout=f)
    subprocess.run(['tar','-xf',str(archive),'-C',str(root)],check=True,timeout=60)
    archive.unlink()
    subprocess.run(['/usr/bin/python3','-B','-m','unittest','discover','-s','tests','-q'],cwd=root,check=True,timeout=300)


def migrate(manifest):
    image=qualification(manifest)
    if command(['docker','image','inspect','--format','{{.Id}}',image],capture=True)!=image:
        raise RuntimeError('Qualified image missing')
    old,before=current()
    if before['version']!='0.13.0-dev':
        raise RuntimeError('Expected running v0.13')
    if before.get('crypto_universe')!=['BTC-EUR','ETH-EUR']:
        raise RuntimeError('Unexpected crypto universe')
    env=ROOT/'.env'
    original=env.read_text()
    updated=env_update(original)
    envmode=env.stat().st_mode & 0o777
    overlay=ROOT/'compose.equity.yaml'
    if overlay.exists() or overlay.is_symlink():
        raise RuntimeError('Overlay already exists')
    unit=UNIT.read_text()
    if 'CacheDirectory=' in unit:
        raise RuntimeError('Research unit already modified')
    if command(['systemctl','is-enabled',TIMER],capture=True)!='enabled':
        raise RuntimeError('Expected enabled research timer')
    command(['systemctl','is-active',TIMER])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    staged=Path(tempfile.mkdtemp(prefix='gta-panel-new-',dir='/opt'))
    staged.chmod(0o755)
    olddir=Path('/opt/gta-equity-pre-panel-'+stamp)
    archive=BASE/('before-panel-'+stamp+'.tar.gz')
    rollback='gta-rollback:panel-'+stamp
    extended=COMPOSE+['-f','compose.equity.yaml']
    up=['up','-d','--no-build','--pull','never','--force-recreate','--wait','--wait-timeout','90','agent']
    prepare(CANDIDATE,staged)
    command(['docker','run','--rm','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','768m','--cpus','1',image,'python','-m','global_agent.preflight'],timeout=180)
    latest,before=current()
    if latest!=old:
        raise RuntimeError('Running image changed')
    minimum=sum(a['fills'] for a in before['accounts'])
    command(['docker','tag',old,rollback])
    # Persist a recovery note without any .env or credential contents.
    recovery=BASE/('panel-recovery-'+stamp+'.json')
    recovery.write_text(json.dumps({'old_image':old,'rollback_tag':rollback,'old_research_dir':str(olddir),'backup':str(archive),'candidate':image,'commit':CANDIDATE},indent=2))
    recovery.chmod(0o600)
    timer_stopped=research_switched=unit_changed=overlay_added=env_changed=crypto_stopped=False
    try:
        command(['systemctl','stop',TIMER])
        timer_stopped=True
        command(['systemctl','stop',SERVICE])
        RESEARCH.rename(olddir)
        research_switched=True
        staged.rename(RESEARCH)
        replace_text(UNIT,(RESEARCH/'ops'/UNIT.name).read_text(),0o644)
        unit_changed=True
        command(['systemctl','daemon-reload'])
        command(['systemd-analyze','verify',SERVICE,TIMER])
        command(['systemctl','start',SERVICE],timeout=660)
        command(probe_args(image,EXPORT),timeout=120)
        replace_text(overlay,(RESEARCH/'compose.equity.yaml').read_text(),0o644)
        overlay_added=True
        # Validate effective mount before stopping the running crypto service.
        cfg=json.loads(command(extended+['config','--format','json'],capture=True))
        svc=cfg['services']['agent']
        if (svc['environment'].get('GTA_EQUITY_REPORT_DIR')!='/equity-report'
                or not any(v.get('source')==str(EXPORT) and v.get('target')=='/equity-report'
                           and v.get('read_only') is True for v in svc['volumes'])):
            raise RuntimeError('Unexpected export mount')
        crypto_stopped=True
        command(COMPOSE+['stop','-t','60','agent'],timeout=90)
        backup_file(archive,old)
        command(['docker','tag',image,TAG])
        replace_text(env,updated,envmode)
        env_changed=True
        command(extended+up,timeout=120)
        wait_panel(minimum)
        cid=command(extended+['ps','-q','agent'],capture=True)
        obj=json.loads(command(['docker','inspect',cid],capture=True))[0]
        if obj['Image']!=image or not any(m.get('Destination')=='/equity-report' and m.get('RW') is False for m in obj['Mounts']):
            raise RuntimeError('Wrong running image or mount')
        command(['systemctl','start',TIMER])
        return {'activated':True,'version':'0.14.0-dev','instruments':28,'backup':str(archive),'recovery':str(recovery)}
    except Exception:
        # Best-effort ordered rollback. No deletion or restoration of live database.
        errors=[]
        def attempt(fn):
            try: fn()
            except Exception as exc: errors.append(type(exc).__name__)
        if crypto_stopped:
            if overlay_added:
                attempt(lambda:command(extended+['stop','-t','60','agent'],timeout=90))
            if env_changed:
                attempt(lambda:replace_text(env,original,envmode))
            attempt(lambda:command(['docker','tag',old,TAG]))
            attempt(lambda:command(COMPOSE+up,timeout=120))
        if overlay_added:
            attempt(lambda:overlay.unlink())
        if research_switched:
            attempt(lambda:command(['systemctl','stop',SERVICE]))
            if RESEARCH.exists():
                attempt(lambda:RESEARCH.rename(staged))
            attempt(lambda:olddir.rename(RESEARCH))
        if unit_changed:
            attempt(lambda:replace_text(UNIT,unit,0o644))
            attempt(lambda:command(['systemctl','daemon-reload']))
        if timer_stopped:
            attempt(lambda:command(['systemctl','start',SERVICE],timeout=660))
            attempt(lambda:command(['systemctl','start',TIMER]))
        if errors:
            raise RuntimeError('Migration failed; rollback incomplete: '+','.join(errors)+'; recovery '+str(recovery))
        raise RuntimeError('Migration failed; previous configuration restarted; recovery '+str(recovery))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--apply',action='store_true',required=True)
    a=p.parse_args()
    if os.geteuid()!=0 or Path.cwd().resolve()!=ROOT:
        p.error('Run as root from /opt/global-trading-agent')
    os.umask(0o077)
    BASE.mkdir(mode=0o700,exist_ok=True)
    with (BASE/'.update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(migrate(a.qualification),indent=2))

if __name__=='__main__':
    main()
