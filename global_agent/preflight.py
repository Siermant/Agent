"""Read-only deployment checks and container resource metrics; no trades."""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from .market import fetch
from .paper_live import load_context, UNIVERSE


def resources(data, cgroup=Path('/sys/fs/cgroup')):
    def number(name):
        try:
            raw = (cgroup / name).read_text().strip()
            return int(raw) if raw.isdigit() else None
        except OSError:
            return None
    try:
        cpu = dict(line.split() for line in (cgroup / 'cpu.stat').read_text().splitlines())
        cpu_usec = int(cpu['usage_usec'])
    except (OSError, ValueError, KeyError):
        cpu_usec = None
    disk = shutil.disk_usage(data)
    return {'memory_bytes': number('memory.current'), 'memory_limit_bytes': number('memory.max'),
            'cpu_usage_usec_total': cpu_usec, 'disk_free_bytes': disk.free, 'disk_total_bytes': disk.total}


def check(quote_source=fetch, context_source=load_context, clock=None):
    clock = clock or (lambda: datetime.now(timezone.utc))
    result = {'real_trading': False, 'checks': {}}
    for name, func in (('history_and_nbp', lambda: context_source(clock().date())), ('current_quotes', quote_source)):
        try:
            value = func()
            if name == 'current_quotes' and not set(UNIVERSE) <= {r[0] for r in value}:
                raise ValueError('Incomplete quotes')
            if name == 'history_and_nbp':
                day, rate, closes = value
                if not 0 < (clock().date()-day).days <= 7 or rate <= 0 or set(closes) != set(UNIVERSE) or any(len(v)!=20 for v in closes.values()):
                    raise ValueError('Incomplete context')
            result['checks'][name] = {'ok': True}
        except Exception as exc:
            result['checks'][name] = {'ok': False, 'error_type': type(exc).__name__}
    result['checked_utc'] = clock().isoformat()
    result['ready'] = all(x['ok'] for x in result['checks'].values())
    return result


def main():
    result = check()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['ready'] else 1)

if __name__ == '__main__':
    main()
