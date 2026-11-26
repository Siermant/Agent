"""Persistent paper-only ledger. No live order or broker API exists here."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sqlite3

D = Decimal
CAPITALS = {"ike": D("28260"), "global": D("50000"), "crypto": D("20000")}
EXPOSURE = {"A": D(".40"), "B": D(".60"), "C": D(".80"), "D": D(".80")}
CENT = D(".01")


def positive(value):
    v = D(str(value))
    if not v.is_finite() or v <= 0:
        raise ValueError("Expected positive finite number")
    return v


def utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timezone-aware timestamp required")
    return value.astimezone(timezone.utc)


class Ledger:
    def __init__(self, path, clock=None, max_age=60):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.max_age = timedelta(seconds=max_age)
        if self.max_age <= timedelta(0):
            raise ValueError("Invalid quote age")
        with self.tx() as db:
            for sql in (
                "CREATE TABLE IF NOT EXISTS accounts(id TEXT PRIMARY KEY, initial TEXT NOT NULL, cash TEXT NOT NULL, peak TEXT NOT NULL, day TEXT, day_equity TEXT, week TEXT, week_equity TEXT)",
                "CREATE TABLE IF NOT EXISTS positions(account TEXT, symbol TEXT, qty TEXT NOT NULL, PRIMARY KEY(account,symbol), FOREIGN KEY(account) REFERENCES accounts(id))",
                "CREATE TABLE IF NOT EXISTS fills(account TEXT, order_id TEXT, symbol TEXT, side TEXT, qty TEXT, price TEXT, fee TEXT, at TEXT, PRIMARY KEY(account,order_id), FOREIGN KEY(account) REFERENCES accounts(id))",
                "CREATE TABLE IF NOT EXISTS snapshots(account TEXT, at TEXT, equity TEXT, PRIMARY KEY(account,at), FOREIGN KEY(account) REFERENCES accounts(id))",
            ):
                db.execute(sql)
            for profile in "ABCDE":
                for kind, capital in CAPITALS.items():
                    db.execute("INSERT OR IGNORE INTO accounts(id,initial,cash,peak) VALUES(?,?,?,?)",
                               (f"{profile}:{kind}", str(capital), str(capital), str(capital)))

    @contextmanager
    def tx(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _account(self, db, account):
        row = db.execute("SELECT * FROM accounts WHERE id=?", (account,)).fetchone()
        if row is None:
            raise ValueError("Unknown account")
        positions = {r['symbol']: D(r['qty']) for r in db.execute("SELECT symbol,qty FROM positions WHERE account=?", (account,))}
        return row, positions

    def _quote(self, quotes, symbol, now):
        # Input: symbol -> (bid PLN, ask PLN, timezone-aware observation time).
        if symbol not in quotes:
            raise ValueError("Missing quote: " + symbol)
        bid, ask, observed = quotes[symbol]
        bid, ask, observed = positive(bid), positive(ask), utc(observed)
        if bid > ask or observed > now or now - observed > self.max_age:
            raise ValueError("Invalid or stale quote: " + symbol)
        return bid, ask

    def _mark(self, positions, quotes, now):
        return sum((qty * self._quote(quotes, sym, now)[0] for sym, qty in positions.items()), D(0))

    def _snapshot(self, db, row, equity, now):
        day = now.date().isoformat()
        week = now.date().isocalendar()
        week = f"{week.year}-{week.week:02d}"
        # Carry last recorded equity into a new UTC day/week; gaps remain visible.
        last = db.execute("SELECT at,equity FROM snapshots WHERE account=? ORDER BY at DESC LIMIT 1", (row['id'],)).fetchone()
        if last and utc(datetime.fromisoformat(last['at'])) > now:
            raise ValueError("Clock moved backwards")
        previous = D(last['equity']) if last else D(row['initial'])
        day_start = D(row['day_equity']) if row['day'] == day else previous
        week_start = D(row['week_equity']) if row['week'] == week else previous
        peak = max(D(row['peak']), equity)
        db.execute("UPDATE accounts SET peak=?,day=?,day_equity=?,week=?,week_equity=? WHERE id=?",
                   (str(peak), day, str(day_start), week, str(week_start), row['id']))
        db.execute("INSERT OR REPLACE INTO snapshots VALUES(?,?,?)", (row['id'], now.isoformat(timespec='microseconds'), str(equity)))
        blocked = (equity <= day_start * D('.98') or equity <= week_start * D('.95') or equity <= peak * D('.90'))
        return blocked, day_start, week_start, peak

    def mark(self, account, quotes):
        with self.tx() as db:
            now = utc(self.clock())
            row, pos = self._account(db, account)
            equity = D(row['cash']) + self._mark(pos, quotes, now)
            blocked, _, _, _ = self._snapshot(db, row, equity, now)
            return {"equity_pln": str(equity), "risk_blocked": blocked}

    def execute(self, account, order_id, symbol, side, qty, quotes, fee_rate, slippage="0", min_fee="0", *, market_open=False, approved=False):
        """Simulated fill from bid/ask. Market eligibility/session supplied by caller.
        Benchmarks use a separate one-time buy and bypass active exposure limits.
        No transfer, leverage, short or real broker functionality.
        """
        if not order_id or not symbol or side not in ('buy', 'sell'):
            raise ValueError("Invalid order")
        qty = positive(qty)
        rate, slip, minimum = map(lambda v: D(str(v)), (fee_rate, slippage, min_fee))
        if any(not v.is_finite() for v in (rate, slip, minimum)) or not (0 <= rate < 1 and 0 <= slip < 1 and minimum >= 0):
            raise ValueError("Invalid costs")
        with self.tx() as db:
            now = utc(self.clock())
            row, pos = self._account(db, account)
            if db.execute("SELECT 1 FROM fills WHERE account=? AND order_id=?", (account, order_id)).fetchone():
                raise ValueError("Duplicate order")
            if not market_open or not approved:
                raise ValueError("Market or eligibility not approved")
            bid, ask = self._quote(quotes, symbol, now)
            price = ask * (1 + slip) if side == 'buy' else bid * (1 - slip)
            value = qty * price
            fee = max(minimum, (value * rate).quantize(CENT, rounding=ROUND_HALF_UP))
            old = pos.get(symbol, D(0))
            cash = D(row['cash'])
            profile = account.split(':', 1)[0]
            if profile == 'E':
                if side != 'buy' or db.execute("SELECT 1 FROM fills WHERE account=?", (account,)).fetchone():
                    raise ValueError("Benchmark allows one initial purchase only")
            if side == 'buy':
                before = cash + self._mark(pos, quotes, now)
                blocked, day_start, week_start, peak = self._snapshot(db, row, before, now)
                cash -= value + fee
                pos[symbol] = old + qty
                invested = self._mark(pos, quotes, now)
                after = cash + invested
                if cash < 0 or after <= 0:
                    raise ValueError("Insufficient cash")
                if profile != 'E' and (blocked or after <= day_start * D('.98') or after <= week_start * D('.95') or after <= peak * D('.90') or cash < after * D('.20') or invested > after * EXPOSURE[profile] or pos[symbol] * bid > after * D('.05')):
                    raise ValueError("Risk limit")
            else:
                if qty > old:
                    raise ValueError("Short prohibited")
                cash += value - fee
                if cash < 0:
                    raise ValueError("Insufficient cash for fees")
                if old == qty:
                    pos.pop(symbol)
                else:
                    pos[symbol] = old - qty
            db.execute("UPDATE accounts SET cash=? WHERE id=?", (str(cash), account))
            db.execute("DELETE FROM positions WHERE account=? AND symbol=?", (account, symbol))
            if symbol in pos:
                db.execute("INSERT INTO positions VALUES(?,?,?)", (account, symbol, str(pos[symbol])))
            db.execute("INSERT INTO fills VALUES(?,?,?,?,?,?,?,?)", (account, order_id, symbol, side, str(qty), str(price), str(fee), now.isoformat(timespec='microseconds')))
            # Reductions remain possible when other quotes are stale; no fake valuation.
            try:
                equity = cash + self._mark(pos, quotes, now)
            except ValueError:
                if side == 'buy':
                    raise
            else:
                self._snapshot(db, self._account(db, account)[0], equity, now)
            return {"order_id": order_id, "price_pln": str(price), "fee_pln": str(fee), "simulated": True}

    def status(self):
        with self.tx() as db:
            out = []
            for row in db.execute("SELECT * FROM accounts ORDER BY id"):
                latest = db.execute("SELECT at,equity FROM snapshots WHERE account=? ORDER BY at DESC LIMIT 1", (row['id'],)).fetchone()
                out.append({"account": row['id'], "initial_pln": row['initial'], "cash_pln": row['cash'], "last_equity_pln": latest['equity'] if latest else None, "valued_at": latest['at'] if latest else None,
                            "positions": self._account(db, row['id'])[1], "fills": db.execute("SELECT count(*) FROM fills WHERE account=?", (row['id'],)).fetchone()[0]})
            return out
