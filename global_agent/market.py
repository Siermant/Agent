"""Read-only OKX EEA spot quote collector; no order submission."""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import sqlite3
from urllib.request import Request, urlopen

URL = "https://eea.okx.com/api/v5/market/tickers?instType=SPOT"
SYMBOLS = frozenset(("BTC-EUR", "ETH-EUR", "SOL-EUR", "XRP-EUR", "ADA-EUR", "DOGE-EUR", "LINK-EUR", "AVAX-EUR"))
MAX_AGE = timedelta(seconds=60)

def now_utc():
    return datetime.now(timezone.utc)

def parse(payload, now):
    if payload.get("code") != "0" or not isinstance(payload.get("data"), list):
        raise ValueError("API rejected request")
    rows = {}
    for item in payload["data"]:
        if not isinstance(item, dict):
            continue
        symbol = item.get("instId")
        if symbol not in SYMBOLS or item.get("instType") != "SPOT":
            continue
        try:
            stamp = datetime.fromtimestamp(int(item["ts"]) / 1000, timezone.utc)
            bid, ask, last = (Decimal(item[k]) for k in ("bidPx", "askPx", "last"))
            if not all(v.is_finite() and v > 0 for v in (bid, ask, last)):
                continue
            if bid > ask or stamp > now or now - stamp > MAX_AGE:
                continue
        except (KeyError, ValueError, TypeError, ArithmeticError, OverflowError, OSError):
            continue
        row = (symbol, stamp.isoformat(timespec="milliseconds"), str(bid), str(ask), str(last))
        if symbol not in rows or row[1] > rows[symbol][1]:
            rows[symbol] = row
    if not rows:
        raise ValueError("No fresh supported EUR spot quotes")
    return list(rows.values())

def fetch():
    req = Request(URL, headers={"Accept": "application/json", "User-Agent": "GlobalTradingAgent/0.3"})
    with urlopen(req, timeout=10) as response:
        raw = response.read(8_000_001)
    if len(raw) > 8_000_000:
        raise ValueError("Response too large")
    return parse(json.loads(raw), now_utc())

class QuoteStore:
    def __init__(self, path):
        self.path = str(path)
        from contextlib import closing
        with closing(self.connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS quotes(symbol TEXT, ts TEXT, bid TEXT, ask TEXT, last TEXT, PRIMARY KEY(symbol,ts))")
            db.execute("CREATE TABLE IF NOT EXISTS feed(id INTEGER PRIMARY KEY CHECK(id=1), ok INTEGER, checked TEXT, reason TEXT)")
    def connect(self):
        # sqlite connection context does not close by itself; use closing in callers
        return sqlite3.connect(self.path, timeout=10)
    def record(self, rows, now):
        from contextlib import closing
        with closing(self.connect()) as db, db:
            for row in rows:
                # Earlier cached responses may arrive after newer ones: keep latest authoritative.
                db.execute("INSERT OR IGNORE INTO quotes VALUES(?,?,?,?,?)", row)
            db.execute("DELETE FROM quotes WHERE ts < ?", ((now-timedelta(days=30)).isoformat(timespec="milliseconds"),))
            db.execute("INSERT OR REPLACE INTO feed VALUES(1,1,?,?)", (now.isoformat(), "OK"))
    def fail(self, now, reason):
        from contextlib import closing
        with closing(self.connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO feed VALUES(1,0,?,?)", (now.isoformat(), str(reason)[:120]))
    def status(self, now):
        from contextlib import closing
        with closing(self.connect()) as db:
            db.row_factory = sqlite3.Row
            feed = db.execute("SELECT * FROM feed WHERE id=1").fetchone()
            healthy = bool(feed and feed["ok"] and timedelta(0) <= now-datetime.fromisoformat(feed["checked"]) <= MAX_AGE)
            items = []
            for row in db.execute("SELECT q.* FROM quotes q JOIN (SELECT symbol,max(ts) AS ts FROM quotes GROUP BY symbol) latest USING(symbol,ts) ORDER BY symbol"):
                item = dict(row)
                item["fresh"] = healthy and timedelta(0) <= now-datetime.fromisoformat(item["ts"]) <= MAX_AGE
                items.append(item)
            return {"source":"OKX EEA", "currency":"EUR", "healthy":healthy and any(x["fresh"] for x in items), "checked":feed["checked"] if feed else None, "reason":feed["reason"] if feed else "Waiting", "quotes":items}
