"""Store Alpaca PAPER credentials locally; never print or overwrite them."""
import getpass
import json
import os
import stat
import sys
import warnings
from pathlib import Path

def main():
    if os.geteuid()!=0 or not sys.stdin.isatty():
        raise SystemExit('Root interactive terminal required')
    warnings.filterwarnings('error',category=getpass.GetPassWarning)
    root=Path('/etc/gta-secrets')
    root.mkdir(mode=0o700,exist_ok=True)
    info=root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid!=0 or info.st_mode & 0o077:
        raise SystemExit('Expected root-only /etc/gta-secrets')
    path=root/'alpaca-paper.json'
    if path.exists() or path.is_symlink():
        raise SystemExit('Credential already exists; no overwrite')
    key=getpass.getpass('Alpaca PAPER API key: ')
    secret=getpass.getpass('Alpaca PAPER secret: ')
    if any(not v or len(v)>1024 or any(c.isspace() for c in v) for v in (key,secret)):
        raise SystemExit('Invalid credential')
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump({'key':key,'secret':secret},f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print('Paper credentials stored with root-only permissions.')

if __name__=='__main__':
    main()
