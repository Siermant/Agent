# Global Trading Agent

**Wersja rozwojowa 0.4.0-dev — wyłącznie monitoring.** Nie wykonuje zleceń, nie posiada integracji z prywatnym API brokera ani automatycznej strategii. Nie uruchamiać starego bitcoin_agent.py. Stary kod i logi pozostają archiwalne; nie należy ich kopiować do obrazu ani używać ujawnionych kluczy.

## Funkcje
- 15 niezależnych rachunków wirtualnych, 491 300 PLN nominalnie.
- Księga SQLite z limitami ryzyka i historią symulowanych wykonań.
- Publiczne notowania spot OKX EEA w EUR, co 30 s; dostępność konkretnych par zależy od źródła.
- Panel po polsku z cenami bid/ask, stanem źródła, saldem i historią ostatniej wyceny.
- Przy utracie połączenia stan źródła jest nieaktualny; nie ma fikcyjnych cen ani transakcji.

## Testy (Python 3.12)
```sh
python3 -m unittest discover -s tests -v
```

## Instalacja próbna na Ubuntu Server 24.04 LTS
Wymagane: Git, Docker Engine z pluginem Compose. Wersja develop służy wyłącznie do testów, nie jest wydaniem produkcyjnym. Przed pierwszym połączeniem ze źródłem zapoznać się z warunkami jego API: https://www.okx.com/en-eu/help/okx-api-agreement.

```sh
git clone --branch develop --single-branch https://github.com/Siermant/Agent.git
cd Agent
docker compose up -d --build
docker compose ps
docker compose logs --tail=50 agent
```

Panel jest publikowany tylko na loopback VM. Z komputera z dostępem do VM (LAN/VPN) uruchomić tunel SSH:
```sh
ssh -N -L 8080:127.0.0.1:8080 UZYTKOWNIK@ADRES_VM
```
Otworzyć http://127.0.0.1:8080. Na VM lokalny test: curl http://127.0.0.1:8080/healthz. Stan healthcheck dotyczy procesu i bazy, nie dostępności giełdy. Nie wystawiać portu publicznie; panel nie ma jeszcze osobnego uwierzytelniania. Docelowo dostęp przez VPN z uwierzytelnianiem aplikacji.

Dane pozostają w nazwanym wolumenie agent-data; nie używać docker compose down -v. Nie kopiować czynnej bazy SQLite zwykłym cp. Wersja testowa nie implementuje jeszcze automatycznej kopii, migracji ani aktualizacji. Obraz bazowy Python 3.12-slim nie jest przypięty do digestu; przed stabilnym wydaniem przypiąć i zweryfikować zależności.

## Kolejne etapy
FX z odrębnym znacznikiem świeżości, dane akcji i ETF, kalendarz sesji, identyfikatory instrumentów, backtest, benchmark i strategie, pełny moduł ryzyka, podatki, uwierzytelnianie, kopie zapasowe. Zobacz ROADMAP.md. Zakup VPS i rzeczywisty handel pozostają odłożone.

## Backtesting v0.5 (osobny tryb badawczy)
Panel monitoringu pozostaje v0.4; ten moduł nie uruchamia automatycznych strategii live.
Przykład pobrania historii w istniejącym kontenerze (po weryfikacji połączenia i warunków API):
```sh
docker compose exec agent python -m global_agent.history --start 2025-01-01 --end 2025-12-31 --symbols BTC-EUR ETH-EUR --output /data/crypto_2025.csv
docker compose exec agent python -m global_agent.backtest /data/crypto_2025.csv --kind crypto --benchmark BTC-EUR
```
Źródła: OKX EEA history-candles, 1Dutc, tylko confirm=1; NBP tabela A EUR, zawsze kurs opublikowany przed dniem świecy. Kurs referencyjny nie jest ceną rzeczywistej wymiany waluty. Brakujące dni powodują błąd, bez interpolacji. Zakres 21–730 pełnych dni; dostępność par i historii zależy od API. Nie uruchamiać wielu pobrań równocześnie.
Alternatywnie dostarczyć CSV date,symbol,open_pln,close_pln, z jednolitym, sprawdzonym kalendarzem sesji i cenami w PLN. Pierwsze 20 sesji stanowi rozgrzewkę, sygnał z poprzednich zamknięć, wykonanie na otwarciu następnej sesji. Każde uruchomienie testuje 5 niezależnych wariantów dla wybranego rodzaju rachunku; nie modyfikuje bazy monitoringu. Wyniki JSON na stdout. Koszty domyślne to założenia: prowizja 0,25%, pełny spread 0,2%, poślizg 0,1% na stronę; parametry konfigurowalne. To nie aktualny cennik giełdy. Akcje domyślnie w całych sztukach, krypto z założonym krokiem 0,000001. D jest regułowy, nie AI. C nie jest jeszcze pełnym gridem. Brak podatków, dywidend, corporate actions, intraday, danych o płynności i rzeczywistych kalendarzy akcji. Testować małe uniwersum; aktywne portfele i benchmark mają różne ekspozycje. Nie interpretować wyniku jako prognozy.
