"""Monitor and optional crypto paper simulation; no broker credentials."""
from contextlib import closing
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import logging
import os
import signal
import threading

from .ledger import Ledger
from .presentation import request_path, equity_chart
from .service_guard import service_lock
from .preflight import resources
from .diagnostics import statistics, report, decision_counts, recent_executions
from .remote_diag import GitHubSink, Outbox, run_cycle
from .backup import daily_backup
from .maintenance import compact_snapshots
from .paper_live import PaperRunner, configured_universe
from .market import QuoteStore, fetch, now_utc

LOG = logging.getLogger("global_agent")
VERSION = "0.12.0-dev"


def poll(store, stop, interval=30, source=fetch, runner=None, runtime=None):
    while not stop.is_set():
        try:
            store.record(source(), now_utc())
        except Exception as exc:
            LOG.warning("Market unavailable: %s", type(exc).__name__)
            store.fail(now_utc(), type(exc).__name__)
            if runtime is not None:
                runtime["paper"] = {"enabled": runner is not None, "state": "feed_unavailable"}
        else:
            if runner is not None:
                try:
                    state = runner.tick()
                except Exception as exc:
                    LOG.warning("Paper cycle blocked: %s", type(exc).__name__)
                    state = "blocked:" + type(exc).__name__
                if runtime is not None:
                    runtime["paper"] = {"enabled": True, "state": state, "checked": now_utc().isoformat()}
        stop.wait(interval)


def backup_loop(data, stop, runtime, ledger=None):
    last_compacted = None
    while not stop.is_set():
        try:
            path = daily_backup(data)
            runtime["backup"] = {"state": "ok", "day": path.name}
            if ledger is not None and last_compacted != path.name:
                compact_snapshots(ledger, now_utc())
                last_compacted = path.name
        except Exception as exc:
            LOG.warning("Backup failed: %s", type(exc).__name__)
            runtime["backup"] = {"state": "failed", "reason": type(exc).__name__}
        stop.wait(300)


def remote_loop(data, ledger, store, stop, runtime, token_path):
    delay = 60
    while not stop.is_set():
        try:
            token = Path(token_path).read_text(encoding="utf-8").strip()
            sink = GitHubSink(token)
            run_cycle(Outbox(data / "diagnostic-outbox"), sink,
                      view(ledger, store, runtime), decision_counts(ledger))
            runtime["remote"] = {"state": "ok", "checked": now_utc().isoformat()}
            delay = 60
        except Exception as exc:
            LOG.warning("Remote diagnostic cycle failed: %s", type(exc).__name__)
            runtime["remote"] = {"state": "pending_or_blocked", "checked": now_utc().isoformat()}
            delay = min(delay * 2, 900)
        stop.wait(delay)


def view(ledger, store, runtime=None):
    paper = (runtime or {}).get("paper", {"enabled": False, "state": "disabled"})
    now = now_utc()
    market = store.status(now)
    paper_ok = bool(paper["enabled"] and paper["state"] == "ok" and paper.get("checked") and market["healthy"] and 0 <= (now-datetime.fromisoformat(paper["checked"])).total_seconds() <= 90)
    return {"version": VERSION, "mode": "crypto-paper" if paper["enabled"] else "monitor-only", "real_trading": False,
            "strategies_running": paper_ok, "paper": paper, "crypto_universe": list(configured_universe()), "time": now.isoformat(),
            "market": market, "accounts": ledger.status(), "statistics": statistics(ledger, now),
            "resources": resources(ledger.path.parent),
            "recent_executions": recent_executions(ledger),
            "backup": (runtime or {}).get("backup", {"state": "waiting"}),
            "remote": (runtime or {}).get("remote", {"state": "disabled"})}


def render(state):
    def cell(value):
        return "<td>" + escape(str(value if value is not None else "—")) + "</td>"
    def row(values):
        return "<tr>" + "".join(cell(v) for v in values) + "</tr>"
    market = state["market"]
    accounts = state["accounts"]
    quotes = "".join(row([q["symbol"], q["bid"], q["ask"], q["last"], q["ts"], "Aktualne" if q["fresh"] else "Nieaktualne"]) for q in market["quotes"])
    portfolios = "".join(row([p["account"], p["initial_pln"], p["cash_pln"], p["last_equity_pln"], p["valued_at"], p["fills"]]) for p in accounts)
    def percent(value):
        return None if value is None else f"{float(value):+.2f}%"
    stats = [x for x in state.get("statistics", []) if x["account"].endswith(":crypto")]
    def stat_row(x):
        fresh = x["fresh"] and market["healthy"] and state["strategies_running"]
        return row([x["account"], percent(x["return_pct"]), percent(x["current_drawdown_pct"]), f"{float(x['fees_pln']):.2f}", x["fills"], "Aktualna" if fresh else ("Brak wyceny" if x["equity_pln"] is None else "Ostatnia / nieaktualna")])
    stats_rows = "".join(stat_row(x) for x in stats)
    chart = equity_chart(stats)
    trades = "".join(row([x["at"], x["account"], x["symbol"], "Kupno" if x["side"]=="buy" else "Sprzedaż", x["quantity"], x["price_pln"], x["fee_pln"]]) for x in state.get("recent_executions", [])[:50])
    backup = state.get("backup", {})
    backup_label = {"ok":"Wykonana", "failed":"Błąd", "waiting":"Oczekiwanie"}.get(backup.get("state"), "Nieznany")
    remote_label = {"ok":"Cykl OK", "pending_or_blocked":"Oczekuje lub zablokowano", "waiting":"Oczekiwanie", "disabled":"Wyłączona"}.get(state.get("remote", {}).get("state"), "Nieznany")
    res = state.get("resources", {})
    def mib(value):
        return "—" if value is None else f"{value / 1048576:.1f} MiB"
    resource_label = "RAM kontenera: " + mib(res.get("memory_bytes")) + " / " + mib(res.get("memory_limit_bytes")) + " | Wolne miejsce na dysku: " + mib(res.get("disk_free_bytes"))
    paper = state["paper"]
    mode = "Symulacja (" + str(len(state.get("crypto_universe", configured_universe()))) + " par EUR, raz na dobę) — " + paper["state"] if paper["enabled"] else "Tryb monitorowania"
    if paper["enabled"] and not state["strategies_running"]:
        mode += " (wstrzymana lub niepotwierdzona aktualność)"
    health = "Dane aktualne" if market["healthy"] else "Brak aktualnych danych — handel zablokowany"
    return """<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30">
<title>Global Trading Agent</title><style>
body{font:16px system-ui,sans-serif;background:#101828;color:#e6edf5;margin:0 auto;max-width:1100px;padding:24px}
h1,h2{font-weight:600}section{background:#1c283a;padding:20px;border-radius:10px;margin:18px 0}
small,p{color:#b8c8da}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:10px;border-bottom:1px solid #34435a;white-space:nowrap} .scroll{overflow:auto}
a{color:#9ed7ff} .status{font-weight:600}
</style></head><body><h1>Global Trading Agent</h1>
<p>""" + escape(mode) + """ • bez handlu rzeczywistego</p>
<section><h2>Rynek kryptowalut — EUR</h2><p class="status">""" + escape(health) + """</p>
<p>Źródło OKX EEA | ostatnia kontrola: """ + escape(str(market["checked"] or "—")) + """ UTC</p>
<div class="scroll"><table><thead><tr><th>Instrument</th><th>Bid</th><th>Ask</th><th>Ostatnia cena</th><th>Czas UTC</th><th>Stan</th></tr></thead><tbody>""" + quotes + """</tbody></table></div></section>
<section><h2>15 portfeli wirtualnych — PLN</h2><p>A: zachowawczy · B: zrównoważony · C: aktywny · D: adaptacyjny · E: benchmark. Akcje i IKE pozostają nieaktywne. Kurs PLN jest referencyjny; wyceny mają znaczniki czasu.</p>
<div class="scroll"><table><thead><tr><th>Portfel</th><th>Kapitał startowy</th><th>Gotówka</th><th>Ostatnia wycena</th><th>Czas UTC</th><th>Transakcje</th></tr></thead><tbody>""" + portfolios + """</tbody></table></div></section>
<section><h2>Porównanie wariantów — krypto</h2><p>Zwrot od startu, bieżące obsunięcie od szczytu i prowizje. Ostatnie wyceny mogą być nieaktualne. Benchmark ma inną ekspozycję, brak podatków.</p>
<div class="scroll"><table><thead><tr><th>Portfel</th><th>Zwrot</th><th>Obsunięcie</th><th>Prowizje PLN</th><th>Transakcje</th><th>Wycena</th></tr></thead><tbody>""" + stats_rows + """</tbody></table></div></section>
<section><h2>Zwrot strategii A–E — ostatnie 30 dni</h2>""" + chart + """</section>
<section><h2>Ostatnie transakcje symulacyjne</h2><div class="scroll"><table><thead><tr><th>Czas UTC</th><th>Portfel</th><th>Instrument</th><th>Strona</th><th>Ilość</th><th>Cena PLN</th><th>Prowizja PLN</th></tr></thead><tbody>""" + trades + """</tbody></table></div></section>
<section><h2>Diagnostyka</h2><p>Kopia lokalna: """ + escape(backup_label + " " + str(backup.get("day", ""))) + """</p>
<p>""" + escape(resource_label) + """</p>
<p>Diagnostyka zdalna: """ + escape(remote_label) + """</p>
<p><a href="/raport.zip">Pobierz raport.zip</a> (wyłącznie dane symulacyjne i status, bez surowych logów i baz).</p></section>
<p>Wersja """ + escape(state["version"]) + """ · <a href="/api/status">JSON</a> · odświeżanie co 30 s</p></body></html>"""


def handler(ledger, store, runtime=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = request_path(self.path)
            if path == "/healthz":
                try:
                    ledger.status()
                    store.status(now_utc())
                except Exception:
                    self.send_error(503)
                    return
                self.respond(b"ok", "text/plain; charset=utf-8")
            elif path in ("/", "/api/status", "/raport.zip"):
                try:
                    state = view(ledger, store, runtime)
                    if path == "/":
                        body, kind = render(state).encode("utf-8"), "text/html; charset=utf-8"
                    elif path == "/raport.zip":
                        body, kind = report(ledger, store, state), "application/zip"
                    else:
                        body, kind = json.dumps(state, default=str).encode(), "application/json"
                except Exception:
                    LOG.exception("Status read failed")
                    self.send_error(503)
                    return
                self.respond(body, kind)
            else:
                self.send_error(404)

        def respond(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if kind == "application/zip":
                self.send_header("Content-Disposition", 'attachment; filename="raport.zip"')
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)
    return Handler


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = Path(os.getenv("GTA_DATA", "./data"))
    data.mkdir(parents=True, exist_ok=True)
    with service_lock(data):
        serve(data)


def serve(data):
    ledger = Ledger(data / "ledger.db")
    store = QuoteStore(data / "market.db")
    enabled = os.getenv("GTA_PAPER_CRYPTO", "0") == "1"
    runner = PaperRunner(ledger, store) if enabled else None
    runtime = {"paper": {"enabled": enabled, "state": "waiting" if enabled else "disabled"}}
    stop = threading.Event()
    interval = max(15, int(os.getenv("GTA_POLL_SECONDS", "30")))
    server = ThreadingHTTPServer((os.getenv("GTA_BIND", "127.0.0.1"), 8080), handler(ledger, store, runtime))
    worker = threading.Thread(target=poll, args=(store, stop, interval), kwargs={"runner": runner, "runtime": runtime}, daemon=True)
    def shutdown(signum, frame):
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    backup_worker = threading.Thread(target=backup_loop, args=(data, stop, runtime, ledger), daemon=True)
    worker.start()
    backup_worker.start()
    token_path = os.getenv("GTA_DIAG_TOKEN_FILE")
    remote_worker = None
    if token_path:
        runtime["remote"] = {"state": "waiting"}
        remote_worker = threading.Thread(target=remote_loop, args=(data, ledger, store, stop, runtime, token_path), daemon=True)
        remote_worker.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=15)
        backup_worker.join(timeout=5)
        if remote_worker is not None:
            remote_worker.join(timeout=5)

if __name__ == "__main__":
    main()
