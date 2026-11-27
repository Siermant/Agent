"""Opt-in daily crypto paper trading, using current quotes and closed bars."""
from datetime import datetime, timedelta
from decimal import Decimal as D, ROUND_DOWN
from contextlib import contextmanager
import fcntl
import os
from functools import partial
from .market import SYMBOLS

from .backtest import signal, WARMUP
from .history import fetch_fx, fetch_symbol, prior_rate
from .ledger import EXPOSURE, positive, utc

UNIVERSE = ('BTC-EUR', 'ETH-EUR')
ALL_UNIVERSE = UNIVERSE + tuple(sorted(SYMBOLS - set(UNIVERSE)))


def configured_universe():
    mode = os.getenv('GTA_CRYPTO_UNIVERSE', 'core')
    if mode == 'core':
        return UNIVERSE
    if mode == 'all':
        return ALL_UNIVERSE
    raise ValueError('Invalid GTA_CRYPTO_UNIVERSE')
FEE = D('.0025')  # research assumptions, not verified exchange tariff
SLIP = D('.001')
LOT = D('.000001')


def load_context(day, universe=UNIVERSE):
    start = day - timedelta(days=WARMUP)
    end = day - timedelta(days=1)
    rates = fetch_fx(start, end)
    ordered = sorted(rates)
    fx_date, fx = prior_rate(day, rates, ordered)
    closes = {}
    for sym in universe:
        rows = fetch_symbol(sym, start, end)
        closes[sym] = [c * prior_rate(d, rates, ordered)[1] for d, (_, c) in sorted(rows.items())]
    return fx_date, fx, closes


class PaperRunner:
    def __init__(self, ledger, store, context=None, universe=None):
        self.ledger = ledger
        self.store = store
        self.universe = tuple(configured_universe() if universe is None else universe)
        if self.universe not in (UNIVERSE, ALL_UNIVERSE):
            raise ValueError('Invalid trading universe')
        self.context = context if context is not None else partial(load_context, universe=self.universe)
        self.cached = None
        with ledger.tx() as db:
            db.execute('CREATE TABLE IF NOT EXISTS paper_decisions(account TEXT, day TEXT, symbol TEXT, result TEXT, at TEXT, PRIMARY KEY(account,day,symbol))')

    @contextmanager
    def lock(self):
        with open(str(self.ledger.path)+'.paper.lock', 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def state(self, account):
        return next(s for s in self.ledger.status() if s['account'] == account)

    def record(self, account, day, sym, result):
        with self.ledger.tx() as db:
            db.execute('INSERT OR IGNORE INTO paper_decisions VALUES(?,?,?,?,?)',
                       (account, str(day), sym, result, utc(self.ledger.clock()).isoformat()))

    def seen(self, account, day, sym, oid):
        with self.ledger.tx() as db:
            if db.execute('SELECT 1 FROM paper_decisions WHERE account=? AND day=? AND symbol=?', (account,str(day),sym)).fetchone():
                return True
            filled = db.execute('SELECT 1 FROM fills WHERE account=? AND order_id=?', (account,oid)).fetchone()
        if filled:
            self.record(account,day,sym,'recovered_fill')
        return bool(filled)

    def tick(self):
        with self.lock():
            return self._tick()

    def _tick(self):
        now = utc(self.ledger.clock())
        day = now.date()
        if self.cached is None or self.cached[0] != day:
            fx_date, fx, closes = self.context(day)
            fx = positive(fx)
            if not timedelta(0) < day-fx_date <= timedelta(days=7):
                raise ValueError('Invalid prior FX date')
            if set(closes) != set(self.universe) or any(len(x) != WARMUP for x in closes.values()):
                raise ValueError('Incomplete closed history')
            closes = {s:[positive(v) for v in xs] for s,xs in closes.items()}
            self.cached = day, fx_date, fx, closes
        _, fx_date, fx, closes = self.cached
        now = utc(self.ledger.clock())
        if now.date() != day:
            return 'day_changed'
        feed = self.store.status(now)
        if not feed['healthy']:
            return 'feed_unavailable'
        quotes = {q['symbol']:(positive(q['bid'])*fx,positive(q['ask'])*fx,datetime.fromisoformat(q['ts']))
                  for q in feed['quotes'] if q['fresh'] and q['symbol'] in self.universe}
        if set(quotes) != set(self.universe):
            return 'incomplete_quotes'
        # Reject an incompatible downgrade before any mark or order.
        if any(not set(s['positions']) <= set(self.universe)
               for s in self.ledger.status() if s['account'].endswith(':crypto')):
            return 'incompatible_positions'
        for p in 'ABCDE':
            account = p+':crypto'
            state = self.state(account)
            if not set(state['positions']) <= set(self.universe):
                raise ValueError('Unexpected existing position')
            self.ledger.mark(account,quotes)
            symbols = ('BTC-EUR',) if p == 'E' else self.universe
            # Sells first, then buys; recover completed order IDs after restart.
            symbols = sorted(symbols,key=lambda s:(s not in state['positions'],s))
            for sym in symbols:
                if utc(self.ledger.clock()).date() != day:
                    return 'day_changed'
                oid = f'daily:{day}:{sym}'
                if self.seen(account,day,sym,oid):
                    continue
                state = self.state(account)
                held = state['positions'].get(sym,D(0))
                if p == 'E' and state['fills']:
                    self.record(account,day,sym,'benchmark_hold')
                    continue
                desired = p == 'E' or signal(p,closes[sym],bool(held))
                if bool(held) == desired:
                    self.record(account,day,sym,'hold' if held else 'no_signal')
                    continue
                if held:
                    side, qty = 'sell', held
                else:
                    side = 'buy'
                    mark = self.ledger.mark(account,quotes)
                    cash, equity = D(state['cash_pln']),D(mark['equity_pln'])
                    if p != 'E' and mark['risk_blocked']:
                        self.record(account,day,sym,'risk_blocked')
                        continue
                    invested = sum((n*quotes[s][0] for s,n in state['positions'].items()),D(0))
                    budget = cash if p == 'E' else min(equity*D('.045'),cash-equity*D('.20'),equity*EXPOSURE[p]-invested)
                    price = quotes[sym][1]*(1+SLIP)
                    qty = (max(D(0),budget-D('.01'))/(price*(1+FEE))/LOT).to_integral_value(rounding=ROUND_DOWN)*LOT
                if qty <= 0:
                    self.record(account,day,sym,'no_budget')
                    continue
                try:
                    self.ledger.execute(account,oid,sym,side,qty,quotes,FEE,SLIP,market_open=True,approved=True)
                except ValueError as exc:
                    if str(exc) not in ('Risk limit','Insufficient cash'):
                        raise
                    self.record(account,day,sym,'risk_blocked')
                else:
                    self.record(account,day,sym,'filled_'+side)
        return 'ok'
