"""Paper-only ledger. No broker connectivity or real execution."""
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field

D = Decimal

def money(v):
    x = D(str(v))
    if not x.is_finite():
        raise ValueError("Non-finite amount")
    return x

@dataclass
class PaperPortfolio:
    cash: Decimal
    positions: dict = field(default_factory=dict)
    completed: set = field(default_factory=set)

    def __post_init__(self):
        self.cash = money(self.cash)
        if self.cash <= 0:
            raise ValueError("Initial cash must be positive")

    def equity(self, marks):
        return self.cash + sum((qty * money(marks[s]) for s, qty in self.positions.items()), D(0))

    def execute(self, order_id, symbol, side, quantity, price_pln, fee_rate, marks, allow_buy=False):
        """Apply an explicitly simulated fill; prices and marks must be PLN.
        Caller must validate freshness/session and portfolio risk before enabling buys.
        In-memory prototype; persistence and concurrency are not yet implemented.
        """
        if not order_id or not symbol or order_id in self.completed:
            raise ValueError("Missing or duplicate order")
        q, p, f = map(money, (quantity, price_pln, fee_rate))
        if q <= 0 or p <= 0 or not D(0) <= f < D(1):
            raise ValueError("Invalid fill")
        if side not in ("buy", "sell"):
            raise ValueError("Invalid side")
        old = self.positions.get(symbol, D(0))
        value = q * p
        fee = (value * f).quantize(D("0.01"), rounding=ROUND_HALF_UP)
        if side == "buy":
            if not allow_buy:
                raise ValueError("Buying disabled until risk approval")
            checked = {s: money(marks[s]) for s in self.positions}
            checked[symbol] = money(marks[symbol])
            if any(v <= 0 for v in checked.values()):
                raise ValueError("Invalid market marks")
            cash = self.cash - value - fee
            qty = old + q
            after = cash + sum((n * checked[s] for s,n in self.positions.items() if s != symbol), D(0)) + qty * checked[symbol]
            if after <= 0 or cash < after * D(".20") or qty * checked[symbol] > after * D(".05"):
                raise ValueError("Cash reserve or position limit")
        else:
            if q > old:
                raise ValueError("Short selling prohibited")
            cash, qty = self.cash + value - fee, old - q
        self.cash = cash
        if qty:
            self.positions[symbol] = qty
        else:
            self.positions.pop(symbol, None)
        self.completed.add(order_id)
        return {"order_id": order_id, "symbol": symbol, "side": side, "quantity": str(q), "price_pln": str(p), "fee_pln": str(fee), "simulated": True}

def initial_portfolios():
    return {name: PaperPortfolio(D(amount)) for name, amount in
            (("ike", "28260"), ("global", "50000"), ("crypto", "20000"))}
