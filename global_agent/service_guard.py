"""Single service per local data directory. Not a distributed lease."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

@contextmanager
def service_lock(data):
    path = Path(data) / '.service.lock'
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another agent owns this data directory') from exc
        yield
