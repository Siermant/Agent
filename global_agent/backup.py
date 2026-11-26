"""Local daily SQLite backups, verified before publication. Linux only."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import json
import os
import shutil
import sqlite3
import tempfile


def copy_db(source, target):
    with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True,timeout=10)) as src:
        with closing(sqlite3.connect(target)) as dest:
            src.backup(dest, pages=256, sleep=.05)
            if dest.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise RuntimeError('Backup integrity failure')
            dest.execute('PRAGMA journal_mode=DELETE')
    os.chmod(target,0o600)


def daily_backup(data, now=None, keep=7):
    if keep < 1:
        raise ValueError('Backup retention must be positive')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError('Timezone-aware date required')
    data = Path(data)
    root = data/'backups'
    root.mkdir(mode=0o700,exist_ok=True)
    day = now.astimezone(timezone.utc).date().isoformat()
    final = root/day
    with open(root/'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.is_dir() and (final/'manifest.json').is_file():
            return final
        tmp = Path(tempfile.mkdtemp(prefix='.partial-',dir=root))
        try:
            for name in ('ledger.db','market.db'):
                copy_db(data/name,tmp/name)
            (tmp/'manifest.json').write_text(json.dumps({'created_utc':now.isoformat(),'databases':['ledger.db','market.db'],'check':'quick_check','scope':'individually consistent snapshots; not an atomic pair'},indent=2),encoding='utf-8')
            os.replace(tmp,final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        # Prune only completed backups with date directory names.
        complete = []
        for p in root.iterdir():
            try:
                valid = datetime.strptime(p.name,'%Y-%m-%d').date().isoformat()==p.name
            except ValueError:
                valid = False
            if valid and p.is_dir() and not p.is_symlink() and (p/'manifest.json').is_file():
                complete.append(p)
        for p in sorted(complete,reverse=True)[keep:]:
            shutil.rmtree(p)
    return final
