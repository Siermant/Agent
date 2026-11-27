"""Guarded v0.14 -> v0.15 dashboard migration after first Global valuation.
No changes to either ledger schema, market strategy, or research service.
"""
import argparse
import fcntl
import json
import os
import re
import time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from ops.update_release import ROOT,BASE,COMPOSE,TAG,VOLUME,command,status,check_state,image_id,backup_file
from ops.migrate_equity_panel import replace_text
from ops.stage_release import REPO,SAFE
from ops.stage_global_panel import CANDIDATE

FILES=('compose.yaml','compose.diagnostics.yaml','compose.equity.yaml')
OLD=COMPOSE+['-f',FILES[-1]]
NEW=OLD+['-f','compose.global.yaml']
UP=['up','-d','--no-build','--pull','never','--force-recreate','--wait','--wait-timeout','90','agent']
MOUNTS={'/equity-report':'/var/cache/gta-equity-export','/global-report':'/var/cache/gta-global-export'}


def next_env(original):
    lines=original.splitlines(keepends=True)
    hits=[i for i,l in enumerate(lines) if l.strip().startswith('COMPOSE_FILE=')]
    if len(hits)!=1 or lines[hits[0]].strip()!='COMPOSE_FILE='+':'.join(FILES):
        raise ValueError('Unexpected existing Compose files')
    lines[hits[0]]='COMPOSE_FILE='+':'.join((*FILES,'compose.global.yaml'))+'\n'
    return ''.join(lines)


def qualification(path,now=None):
    m=json.loads(Path(path).read_text())
    now=now or datetime.now(timezone.utc)
    if (m.get('commit')!=CANDIDATE or m.get('activated') is not False
        or any(m.get(k)!='passed' for k in ('unit_tests','offline_http_smoke','global_fixture_http','existing_equity_mount'))
        or not timedelta(0)<=now-datetime.fromisoformat(m['checked_utc'])<=timedelta(hours=24)):
        raise ValueError('Fresh v0.15 qualification required')
    return image_id(m['image_id'])


def panel_ok(s,minimum,version):
    check_state(s,version,minimum,ready=True)
    eq=s['equity_research']
    if eq.get('state')!='ok' or eq.get('instruments')!=28:
        raise RuntimeError('Equity ranking missing')
    if version=='0.15.0-dev':
        g=s['global_paper']
        if (g.get('real_trading') is not False or g.get('simulated') is not True
            or g.get('state') not in ('valued','market_closed','incomplete_quotes','session_changed')
            or len(g.get('accounts',[]))!=5 or sum(r['fills'] for r in g['accounts'])<1):
            raise RuntimeError('Global valuation unavailable')


def running(compose,version,minimum=0):
    cfg=json.loads(command(compose+['config','--format','json'],capture=True))
    svc=cfg['services']['agent']
    env=svc['environment']
    if (cfg.get('name')!='global-trading-agent' or str(env.get('GTA_PAPER_CRYPTO'))!='1'
        or env.get('GTA_DIAG_TOKEN_FILE')!='/run/secrets/gta_diagnostics_token'
        or svc.get('image') not in (None,TAG,TAG.removesuffix(':latest'))):
        raise ValueError('Unexpected Compose service')
    required=('/equity-report',) if version=='0.14.0-dev' else tuple(MOUNTS)
    for target in required:
        if not any(v.get('source')==MOUNTS[target] and v.get('target')==target and v.get('read_only') is True for v in svc['volumes']):
            raise ValueError('Missing read-only report mount')
    cid=command(compose+['ps','-q','agent'],capture=True)
    if not re.fullmatch('[0-9a-f]{64}',cid):
        raise ValueError('Expected one container')
    obj=json.loads(command(['docker','inspect',cid],capture=True))[0]
    if (not obj['State']['Running'] or not any(m.get('Name')==VOLUME and m['Destination']=='/data' for m in obj['Mounts'])
        or any(not any(m['Destination']==t and m['RW'] is False for m in obj['Mounts']) for t in required)):
        raise ValueError('Unexpected actual mounts')
    s=status()
    panel_ok(s,minimum,version)
    return image_id(obj['Image']),s


def actual_reports(image):
    # No private state directory, no token, no network, no crypto data.
    code=('from global_agent.equity_panel import summary as e; '
          'from global_agent.global_panel import summary as g; '
          'a=e("/equity-report"); b=g("/global-report"); '
          'assert a["state"]=="ok" and a["instruments"]==28; '
          'assert b["state"] in ("valued","market_closed","incomplete_quotes","session_changed"); '
          'assert b["real_trading"] is False and len(b["accounts"])==5; '
          'assert sum(x["fills"] for x in b["accounts"])>0')
    command(['docker','run','--rm',*SAFE,
             '--mount','type=bind,src=/var/cache/gta-equity-export,dst=/equity-report,readonly',
             '--mount','type=bind,src=/var/cache/gta-global-export,dst=/global-report,readonly',
             image,'python','-c',code])


def migrate(path):
    image=qualification(path)
    if command(['docker','image','inspect','--format','{{.Id}}',image],capture=True)!=image:
        raise ValueError('Qualified image unavailable')
    if command(['systemctl','is-enabled','gta-global-mark.timer'],capture=True)!='enabled':
        raise ValueError('Persistent valuation timer must be installed first')
    command(['systemctl','is-active','gta-global-mark.timer'])
    old,before=running(OLD,'0.14.0-dev')
    if old==image:
        raise ValueError('Unexpected identical images')
    env=ROOT/'.env'
    original=env.read_text()
    updated=next_env(original)
    mode=env.stat().st_mode&0o777
    overlay=ROOT/'compose.global.yaml'
    if overlay.exists() or overlay.is_symlink():
        raise ValueError('Global overlay already exists')
    command(['git','fetch','--depth=1','--no-tags',REPO,CANDIDATE])
    if command(['git','rev-parse','FETCH_HEAD'],capture=True)!=CANDIDATE:
        raise ValueError('Commit mismatch')
    for name in FILES:
        if command(['git','show',CANDIDATE+':'+name],capture=True)!=(ROOT/name).read_text().strip():
            raise ValueError('Existing Compose differs')
    new_overlay=command(['git','show',CANDIDATE+':compose.global.yaml'],capture=True)+'\n'
    actual_reports(image)
    command(['docker','run','--rm','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','768m','--cpus','1',image,'python','-m','global_agent.preflight'],timeout=180)
    latest,before=running(OLD,'0.14.0-dev')
    if latest!=old:
        raise RuntimeError('Image changed during checks')
    minimum=sum(r['fills'] for r in before['accounts'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    archive=BASE/('before-global-panel-'+stamp+'.tar.gz')
    old_env=BASE/('before-global-panel-'+stamp+'.env')
    replace_text(old_env,original,0o600)
    tag='gta-rollback:global-'+stamp
    command(['docker','tag',old,tag])
    recovery=BASE/('global-panel-recovery-'+stamp+'.json')
    replace_text(recovery,json.dumps({'old_image':old,'rollback_tag':tag,'old_env':str(old_env),
                                    'backup':str(archive),'candidate':image},indent=2),0o600)
    added=stopping=changed=False
    try:
        replace_text(overlay,new_overlay,0o644)
        added=True
        command(NEW+['config','--quiet'])
        stopping=True
        command(OLD+['stop','-t','60','agent'],timeout=90)
        backup_file(archive,old)
        command(['docker','tag',image,TAG])
        replace_text(env,updated,mode)
        changed=True
        command(NEW+UP,timeout=120)
        for i in range(30):
            try:
                actual,_=running(NEW,'0.15.0-dev',minimum)
                if actual!=image:raise RuntimeError('Wrong candidate image')
                break
            except Exception:
                if i==29:raise
                time.sleep(3)
        return {'activated':True,'version':'0.15.0-dev','backup':str(archive),'recovery':str(recovery)}
    except Exception:
        errors=[]
        def attempt(fn):
            try:fn()
            except Exception as exc:errors.append(type(exc).__name__)
        if stopping:
            attempt(lambda:command(NEW+['stop','-t','60','agent'],timeout=90))
            if changed:attempt(lambda:replace_text(env,original,mode))
            attempt(lambda:command(['docker','tag',old,TAG]))
            attempt(lambda:command(OLD+UP,timeout=120))
            attempt(lambda:running(OLD,'0.14.0-dev',minimum))
        if added:attempt(lambda:overlay.unlink())
        if errors:
            raise RuntimeError('Migration failed; rollback incomplete: '+str(recovery))
        raise RuntimeError('Migration failed; previous configuration restored: '+str(recovery))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--apply',action='store_true',required=True)
    a=p.parse_args()
    if os.geteuid()!=0 or Path.cwd().resolve()!=ROOT:
        p.error('Root and live repository required')
    os.umask(0o077)
    BASE.mkdir(mode=0o700,exist_ok=True)
    with (BASE/'.update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(migrate(a.qualification),indent=2))
if __name__=='__main__':
    main()
