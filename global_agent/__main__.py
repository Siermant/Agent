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
from .paper_live import PaperRunner
from .market import QuoteStore, fetch, now_utc

LOG = logging.getLogger("global_agent")
VERSION = "0.6.0-dev"


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


def view(ledger, store, runtime=None):
    paper = (runtime or {}).get("paper", {"enabled": False, "state": "disabled"})
    return {"version": VERSION, "mode": "crypto-paper" if paper["enabled"] else "monitor-only", "real_trading": False,
            "strategies_running": paper["enabled"] and paper["state"] == "ok", "paper": paper, "time": now_utc().isoformat(),
            "market": store.status(now_utc()), "accounts": ledger.status()}


def render(state):
    def cell(value):
        return "<td>" + escape(str(value if value is not None else "—")) + "</td>"
    def row(values):
        return "<tr>" + "".join(cell(v) for v in values) + "</tr>"
    market = state["market"]
    accounts = state["accounts"]
    quotes = "".join(row([q["symbol"], q["bid"], q["ask"], q["last"], q["ts"], "Aktualne" if q["fresh"] else "Nieaktualne"]) for q in market["quotes"])
    portfolios = "".join(row([p["account"], p["initial_pln"], p["cash_pln"], p["last_equity_pln"], p["valued_at"], p["fills"]]) for p in accounts)
    paper = state["paper"]
    mode = "Symulacja BTC/ETH (raz na dobę) — " + paper["state"] if paper["enabled"] else "Tryb monitorowania"
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
<p>Wersja """ + escape(state["version"]) + """ · <a href="/api/status">JSON</a> · odświeżanie co 30 s</p></body></html>"""


def handler(ledger, store, runtime=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                try:
                    ledger.status()
                    store.status(now_utc())
                except Exception:
                    self.send_error(503)
                    return
                self.respond(b"ok", "text/plain; charset=utf-8")
            elif self.path in ("/", "/api/status"):
                try:
                    state = view(ledger, store, runtime)
                    if self.path == "/":
                        body, kind = render(state).encode("utf-8"), "text/html; charset=utf-8"
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
    worker.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=15)

if __name__ == "__main__":
    main()
