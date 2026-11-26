"""Independent paper experiments. No live trading or strategy execution."""
from dataclasses import dataclass
from decimal import Decimal
from .paper import initial_portfolios

@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    strategies: tuple[str, ...]
    target_exposure: Decimal
    enabled: bool = False

D = Decimal
PROFILES = (
    Profile("A", "Conservative", ("trend",), D("0.40")),
    Profile("B", "Balanced", ("trend", "mean_reversion", "breakout"), D("0.60")),
    Profile("C", "Volatility", ("breakout", "momentum", "dynamic_grid"), D("0.80")),
    Profile("D", "Adaptive", ("trend", "mean_reversion", "breakout", "momentum", "dynamic_grid"), D("0.80")),
)

# Targets are experimental settings, not enforced risk limits.
# Strategies remain disabled until the execution and risk engines are implemented.
REFERENCE = {
    "ike": "Eligible global equity UCITS ETF, buy-and-hold",
    "global": "Global equity UCITS ETF, buy-and-hold",
    "crypto": "BTC spot buy-and-hold",
}

def new_experiments():
    """Create isolated portfolio instances, with identical capital per account type."""
    return {profile.key: initial_portfolios() for profile in PROFILES}

def reference_capitals():
    """Reference accounts are calculated separately, not through active trading limits."""
    return {key: p.cash for key, p in initial_portfolios().items()}
