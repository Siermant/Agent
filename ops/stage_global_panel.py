"""Qualify v0.15 against an isolated synthetic Global report, not live holdings."""
import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from ops.stage_release import qualify, command, SAFE, approved_sha
from ops.update_release import BASE

CANDIDATE='2d6ea19830322e9f02d9f960f73bbadd40a2d99a'

def fixture(root):
    from global_agent.global_panel import FLAGS,export_panel,ACCOUNTS
    now=datetime.now(timezone.utc)
    from datetime import timedelta
    at=now.isoformat()
    accounts=[dict(account=a,capital_pln='50000',cash_pln='50000',invested_pln='0',
                   equity_pln='50000',pnl_pln='0',return_pct='0',drawdown_pct='0',
                   risk_blocked=False,fills=0,positions_count=0,positions=[]) for a in ACCOUNTS]
    v=dict(FLAGS,state='valued',checked_utc=at,valued_at=at,quote_source='alpaca_iex_latest',
           fx_date=str(now.date()-timedelta(days=1)),accounts=accounts,
           history=[{'at':at,'equity_pln':{a:'50000' for a in ACCOUNTS}}],fills=[])
    export_panel(root,v,v)

def stage(output):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new manifest path')
    os.umask(0o022)
    result=qualify(CANDIDATE,output.with_name(output.stem+'-image.json'))
    with tempfile.TemporaryDirectory(prefix='gta-global-fixture-') as tmp:
        root=Path(tmp)
        root.chmod(0o755)
        fixture(root)
        cid=command(['docker','run','-d',*SAFE,'--tmpfs',
            '/data:rw,nosuid,nodev,size=128m,uid=10001,gid=10001,mode=0700',
            '-e','GTA_PAPER_CRYPTO=0','-e','GTA_GLOBAL_REPORT_DIR=/global-report',
            '-e','GTA_EQUITY_REPORT_DIR=/equity-report',
            '--mount',f'type=bind,src={root},dst=/global-report,readonly',
            '--mount','type=bind,src=/var/cache/gta-equity-export,dst=/equity-report,readonly',
            result['image_id']],capture=True).strip()
        try:
            probe=('import json,urllib.request as u; '
                   's=json.load(u.urlopen("http://127.0.0.1:8080/api/status",timeout=3)); '
                   'assert s["version"]=="0.15.0-dev" and s["real_trading"] is False; '
                   'assert len(s["accounts"])==15; '
                   'assert s["equity_research"]["state"]=="ok" and s["equity_research"]["instruments"]==28; '
                   'assert s["global_paper"]["state"]=="valued" and len(s["global_paper"]["accounts"])==5; '
                   'h=u.urlopen("http://127.0.0.1:8080/",timeout=3).read().decode(); '
                   'assert "Global A" in h and "Historia zwrotu" in h')
            for i in range(15):
                try:
                    command(['docker','exec',cid,'python','-c',probe],timeout=10)
                    break
                except Exception:
                    if i==14:raise
                    time.sleep(2)
        finally:
            command(['docker','rm','-f',cid])
    result.update(global_fixture_http='passed',existing_equity_mount='passed',
                  checked_utc=datetime.now(timezone.utc).isoformat())
    os.umask(0o077)
    with output.open('x') as f:
        json.dump(result,f,indent=2)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.geteuid()!=0:p.error('Root required on VM 106')
    BASE.mkdir(mode=0o700,exist_ok=True)
    print(json.dumps(stage(a.output),indent=2))
if __name__=='__main__':
    main()
