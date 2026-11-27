"""Qualify v0.14 dashboard bridge on VM 106 without changing either live service."""
import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from ops.stage_release import qualify, approved_sha, SAFE, REPO

STATE='gta-equity-panel-qualify'
CACHE='gta-equity-panel-qualify-export'


def run(args, cwd=None, timeout=120, capture=False):
    return subprocess.run(args,cwd=cwd,timeout=timeout,check=True,
                          text=True,stdout=subprocess.PIPE if capture else None).stdout


def probe_args(image, export):
    probe=('import json; from global_agent.equity_panel import summary,render_equities; '
           's=summary("/equity-report"); '
           'assert s["state"]=="ok" and s["instruments"]==28 and len(s["cohorts"])==2; '
           'h=render_equities(s); assert "WMT" in h and "IBM" in h; '
           'print(json.dumps({"state":s["state"],"instruments":s["instruments"],"cohorts":len(s["cohorts"])}))')
    return ['docker','run','--rm',*SAFE,'--mount',
            f'type=bind,src={export},dst=/equity-report,readonly',
            image,'python','-c',probe]


def stage(sha, output):
    sha=approved_sha(sha)
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use new manifest path')
    # Build and qualify without the live data volume or token.
    qualified=qualify(sha,output.with_name(output.stem+'-image.json'))
    image=qualified['image_id']
    with tempfile.TemporaryDirectory(prefix='gta-equity-panel-',dir='/opt') as tmp:
        root=Path(tmp)
        root.chmod(0o755)
        run(['git','init','--quiet'],cwd=root)
        run(['git','fetch','--depth=1','--no-tags',REPO,sha],cwd=root)
        if run(['git','rev-parse','FETCH_HEAD'],cwd=root,capture=True).strip()!=sha:
            raise ValueError('Commit mismatch')
        run(['git','checkout',sha,'--','global_agent'],cwd=root)
        run(['systemd-run','--wait','--pipe','--collect',
             '-p','DynamicUser=yes','-p','WorkingDirectory='+str(root),
             '-p','StateDirectory='+STATE,'-p','StateDirectoryMode=0700',
             '-p','CacheDirectory='+CACHE,'-p','CacheDirectoryMode=0755',
             '-p','LoadCredential=alpaca-paper.json:/etc/gta-secrets/alpaca-paper.json',
             '-p','UMask=0077','-p','ProtectSystem=strict','-p','ProtectHome=yes',
             '-p','NoNewPrivileges=yes','-p','PrivateTmp=yes',
             '-p','MemoryMax=192M','-p','CPUQuota=50%','-p','RuntimeMaxSec=600',
             '/usr/bin/python3','-B','-s','-m','global_agent.equity_job'],timeout=660)
        export=Path('/var/cache')/CACHE
        # Only this isolated export is mounted; Docker daemon resolves the cache symlink.
        summary=json.loads(run(probe_args(image,export),capture=True))
    result={'commit':sha,'image_id':image,'checked_utc':datetime.now(timezone.utc).isoformat(),
            'offline_qualification':'passed','isolated_provider_scan':'passed',
            'container_export_read':'passed','instruments':summary['instruments'],
            'cohorts':summary['cohorts'],'activated':False}
    with output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approved-sha',required=True,type=approved_sha)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if os.geteuid()!=0:
        p.error('Root required on VM 106')
    # Qualification checkouts must be readable by the isolated service UID.
    os.umask(0o022)
    print(json.dumps(stage(a.approved_sha,a.output),indent=2))

if __name__=='__main__':
    main()
