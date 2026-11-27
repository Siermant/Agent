"""Install isolated Global valuation only after the initial simulated purchases.
Optional panel migration uses a previously qualified, exact candidate image.
No order execution, no schema change, no resetting the experiment.
"""
import argparse
import fcntl
import json
import os
import shutil
import tempfile
from datetime import datetime,timezone
from pathlib import Path
from ops.update_release import BASE,ROOT,command
from ops.migrate_global_panel import qualification,migrate,running,OLD,actual_reports
from ops.migrate_equity_panel import replace_text
from global_agent.global_panel import summary

SOURCE=Path(__file__).resolve().parents[1]
DEST=Path('/opt/gta-global-mark')
UNIT_ROOT=Path('/etc/systemd/system')
LEDGER=Path('/var/lib/gta-global-paper/global-paper.db')
EXPORT=Path('/var/cache/gta-global-export')
SERVICE='gta-global-mark.service'
TIMER='gta-global-mark.timer'
INITIAL='gta-global-initial.timer'
INITIAL_SERVICE='gta-global-initial.service'

def check_report(directory=EXPORT):
    report=summary(directory)
    if (report.get('state')!='valued' or report.get('fresh') is not True
        or len(report.get('accounts',[]))!=5
        or sum(r['fills'] for r in report['accounts'])<1):
        raise RuntimeError('First fresh Global valuation not available; no panel migration')
    return report

def unit_state(name):
    return command(['systemctl','show',name,'-p','ActiveState','--value'],capture=True)

def install(qualification_path):
    # Nothing changes before first opening. No database is created here.
    if not LEDGER.is_file():
        return {'state':'waiting_for_initial_open','activated':False}
    image=qualification(qualification_path)
    if command(['docker','image','inspect','--format','{{.Id}}',image],capture=True)!=image:
        raise RuntimeError('Qualified image unavailable')
    running(OLD,'0.14.0-dev')
    if any((UNIT_ROOT/n).exists() or (UNIT_ROOT/n).is_symlink() for n in (SERVICE,TIMER)):
        raise RuntimeError('Valuation service already installed; inspect status instead of reinstalling')
    if DEST.is_symlink() or (DEST.exists() and not DEST.is_dir()):
        raise RuntimeError('Unexpected valuation source directory')
    if unit_state(INITIAL_SERVICE) in ('active','activating','deactivating','reloading'):
        raise RuntimeError('Initial opening service busy; retry when finished')
    command(['systemd-analyze','verify',str(SOURCE/'ops'/SERVICE),str(SOURCE/'ops'/TIMER)])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    temp=Path(tempfile.mkdtemp(prefix='gta-mark-new-',dir='/opt'))
    temp.chmod(0o755)
    previous=DEST.with_name('gta-global-mark-pre-panel-'+stamp)
    moved=placed=False
    installed=[]
    try:
        for name in ('global_agent','tests','ops'):
            shutil.copytree(SOURCE/name,temp/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        # systemd DynamicUser needs access to source, never credentials.
        for p in temp.rglob('*'):
            p.chmod(0o755 if p.is_dir() else 0o644)
        if DEST.exists():
            DEST.rename(previous)
            moved=True
        temp.rename(DEST)
        placed=True
        for name in (SERVICE,TIMER):
            target=UNIT_ROOT/name
            replace_text(target,(DEST/'ops'/name).read_text(),0o644)
            installed.append(target)
        command(['systemctl','daemon-reload'])
        command(['systemctl','start',SERVICE],timeout=180)
        report=check_report()
        actual_reports(image)
        command(['systemctl','enable','--now',TIMER])
        command(['systemctl','is-active',TIMER])
        command(['systemctl','is-enabled',TIMER])
    except Exception:
        # Do not remove valuation DB, cache, or the original experiment.
        errors=[]
        def attempt(fn):
            try:fn()
            except Exception as exc:errors.append(type(exc).__name__)
        if installed:
            attempt(lambda:command(['systemctl','disable','--now',TIMER]))
            attempt(lambda:command(['systemctl','stop',SERVICE]))
            for target in installed:attempt(lambda target=target:target.unlink())
        if placed:attempt(lambda:DEST.rename(temp))
        if moved:attempt(lambda:previous.rename(DEST))
        attempt(lambda:command(['systemctl','daemon-reload']))
        if errors:raise RuntimeError('Mark installation failed; rollback incomplete: '+','.join(errors))
        raise
    # Initial experiment now exists and has validated marks. No new opening is
    # needed; stop only the transient timer if still active, not the ledger.
    if unit_state(INITIAL)=='active':
        command(['systemctl','stop',INITIAL])
    return {'state':'valuation_active','activated':False,'valued_at':report['valued_at'],
            'fills':sum(r['fills'] for r in report['accounts']),
            'previous_source':str(previous) if moved else None}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',required=True,type=Path)
    p.add_argument('--apply-panel',action='store_true')
    a=p.parse_args()
    if os.geteuid()!=0 or Path.cwd().resolve()!=ROOT:
        p.error('Root and live repository required')
    os.umask(0o077)
    BASE.mkdir(mode=0o700,exist_ok=True)
    with (BASE/'.update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (UNIT_ROOT/TIMER).is_file():
            # Resume after successful valuation installation, failed panel update.
            command(['systemctl','is-enabled',TIMER])
            command(['systemctl','is-active',TIMER])
            result={'state':'valuation_already_installed','activated':False}
        else:
            result=install(a.qualification)
        print(json.dumps(result,indent=2),flush=True)
        if a.apply_panel and result['state']!='waiting_for_initial_open':
            print(json.dumps(migrate(a.qualification),indent=2))
if __name__=='__main__':
    main()
