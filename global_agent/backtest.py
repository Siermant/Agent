"""Offline daily-bar research. No broker or network access.
CSV prices must already be in PLN, aligned on a verified session calendar.
Signals use prior closes only; fills use the next session open.
"""
import argparse
import csv
import json
import tempfile
from collections import defaultdict
from datetime import date, datetime, time, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from statistics import mean
from .ledger import Ledger, CAPITALS, EXPOSURE, positive

D = Decimal
WARMUP = 20


def load_csv(path):
    days = defaultdict(dict)
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'date', 'symbol', 'open_pln', 'close_pln'} <= set(reader.fieldnames or ()):
            raise ValueError('Expected date,symbol,open_pln,close_pln')
        for row in reader:
            day = date.fromisoformat(row['date'])
            symbol = row['symbol'].strip()
            if not symbol or symbol in days[day]:
                raise ValueError('Empty symbol or duplicate bar')
            days[day][symbol] = (positive(row['open_pln']), positive(row['close_pln']))
    dates = sorted(days)
    if len(dates) <= WARMUP:
        raise ValueError('Need at least 21 aligned sessions')
    symbols = set(days[dates[0]])
    if any(set(days[d]) != symbols for d in dates):
        raise ValueError('Missing bars: use a common verified session calendar')
    return [(day, days[day]) for day in dates]


def signal(profile, closes, held):
    """Only closed sessions supplied. D is rules-based, not a trained AI."""
    if len(closes) < WARMUP:
        return False
    window = closes[-20:]
    last = window[-1]
    avg = mean(window)
    fast = mean(window[-5:])
    trend = last > avg and fast > avg
    reversal = last < avg * D('.97')
    breakout = last > max(window[:-1])
    momentum = last > window[-6] * D('1.02')
    grid = last < avg * D('.98')
    if held:
        if profile == 'A':
            return last >= avg
        if profile == 'B':
            return avg * D('.95') <= last <= avg * D('1.08') or trend
        if profile == 'C':
            return avg * D('.97') <= last <= avg * D('1.06') and (momentum or last < avg)
        return last >= avg * D('.97') and (trend or last < avg)
    if profile == 'A':
        return trend
    if profile == 'B':
        return trend or reversal or breakout
    if profile == 'C':
        return breakout or momentum or grid
    if profile == 'D':
        return (trend or breakout) if fast > avg else reversal
    raise ValueError('Unknown profile')


def run(bars, kind, benchmark, fee='0.0025', spread='0.002', slippage='0.001', min_fee='0', lot=None):
    """A-E over one aligned market/account type. Costs are user assumptions."""
    if kind not in CAPITALS or benchmark not in bars[0][1]:
        raise ValueError('Invalid account type or benchmark')
    fee, spread, slip, minimum = map(lambda x: D(str(x)), (fee, spread, slippage, min_fee))
    if any(not x.is_finite() for x in (fee, spread, slip, minimum)) or not (0 <= fee < 1 and 0 <= spread < 1 and 0 <= slip < 1 and minimum >= 0):
        raise ValueError('Invalid cost assumptions')
    step = positive(lot if lot is not None else ('0.000001' if kind == 'crypto' else '1'))
    def round_qty(q):
        return (q / step).to_integral_value(rounding=ROUND_DOWN) * step
    clock = [None]
    closes = defaultdict(list)
    history = {p: [] for p in 'ABCDE'}
    blocked = {p: 0 for p in 'ABCDE'}
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(Path(tmp) / 'backtest.db', clock=lambda: clock[0])
        def states():
            return {p['account'].split(':')[0]: p for p in ledger.status() if p['account'].endswith(':'+kind)}
        def quotes(values, idx):
            return {s: (v[idx]*(1-spread/2), v[idx]*(1+spread/2), clock[0]) for s,v in values.items()}
        def fill(p, sym, side, qty, q, day):
            if qty <= 0:
                return False
            try:
                ledger.execute(p+':'+kind, f'{day}:{sym}:{side}', sym, side, qty, q, fee, slip, minimum, market_open=True, approved=True)
            except ValueError as exc:
                if str(exc) not in ('Risk limit', 'Insufficient cash'):
                    raise
                blocked[p] += 1
                return False
            return True
        for index, (day, values) in enumerate(bars):
            if index >= WARMUP:
                clock[0] = datetime.combine(day, time(9), timezone.utc)
                q = quotes(values, 0)
                # These timestamps represent abstract session phases, not actual exchange hours.
                for p, state in states().items():
                    positions = state['positions']
                    if p == 'E':
                        if not state['fills']:
                            ask = q[benchmark][1]*(1+slip)
                            available = D(state['cash_pln'])-minimum-D('.01')
                            qty = round_qty(max(D(0), available)/(ask*(1+fee)))
                            fill(p, benchmark, 'buy', qty, q, day)
                        continue
                    # Sell first, then use cash from completed simulated fills.
                    for sym, qty in sorted(positions.items()):
                        if not signal(p, closes[sym], True):
                            fill(p, sym, 'sell', qty, q, day)
                    for sym in sorted(values):
                        current = states()[p]
                        if sym in current['positions'] or not signal(p, closes[sym], False):
                            continue
                        mark = ledger.mark(p+':'+kind, q)
                        if mark['risk_blocked']:
                            blocked[p] += 1
                            break
                        equity = D(mark['equity_pln'])
                        cash = D(current['cash_pln'])
                        invested = sum((n*q[s][0] for s,n in current['positions'].items()), D(0))
                        budget = min(equity*D('.045'), cash-equity*D('.20'), equity*EXPOSURE[p]-invested)
                        ask = q[sym][1]*(1+slip)
                        qty = round_qty(max(D(0), budget-minimum-D('.01'))/(ask*(1+fee)))
                        fill(p, sym, 'buy', qty, q, day)
                clock[0] = datetime.combine(day, time(17), timezone.utc)
                q = quotes(values, 1)
                for p in 'ABCDE':
                    equity = D(ledger.mark(p+':'+kind, q)['equity_pln'])
                    history[p].append((day.isoformat(), equity))
            for sym, (_, close) in values.items():
                closes[sym].append(close)
        result = {}
        for p, state in states().items():
            peak = CAPITALS[kind]
            drawdown = D(0)
            for _, equity in history[p]:
                peak = max(peak, equity)
                drawdown = max(drawdown, 1-equity/peak)
            end = history[p][-1][1]
            result[p] = {'equity_pln': str(end), 'return_pct': str((end/CAPITALS[kind]-1)*100), 'max_close_drawdown_pct': str(drawdown*100), 'fills': state['fills'], 'blocked': blocked[p], 'history': history[p]}
    return {'mode':'offline_backtest', 'account_type':kind, 'benchmark':benchmark, 'warmup_sessions':WARMUP, 'assumptions':{'fee':str(fee),'spread':str(spread),'slippage':str(slip),'min_fee':str(minimum),'lot':str(step)}, 'limitations':['PLN input, externally verified aligned sessions and eligibility','No taxes, dividends, corporate actions, intraday stops or liquidity model','D is rules-based; C is not yet a true grid','Benchmark has higher exposure than active profiles; compare risk as well as returns'], 'profiles':result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--kind', choices=CAPITALS, required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--fee', default='0.0025')
    parser.add_argument('--spread', default='0.002')
    parser.add_argument('--slippage', default='0.001')
    parser.add_argument('--min-fee', default='0')
    parser.add_argument('--lot')
    args = parser.parse_args()
    print(json.dumps(run(load_csv(args.csv), args.kind, args.benchmark, args.fee, args.spread, args.slippage, args.min_fee, args.lot), default=str, indent=2))

if __name__ == '__main__':
    main()
