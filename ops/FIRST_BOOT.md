# Pierwsza instalacja testowa — Proxmox (27.11.2026)
Nie jest to wydanie produkcyjne. Brak rzeczywistych transakcji. Najpierw monitoring, następnie paper trading po potwierdzeniu poprawności źródeł i zapisu.

## 1. Host Proxmox — tylko odczyt
W konsoli hosta:
```sh
lscpu
free -h
pvesm status
qm list
pct list
```
Zależnie od wolnych zasobów utworzyć osobną VM: Ubuntu Server 24.04 LTS, 2 vCPU host, 3 GB RAM, 40 GB SSD, VirtIO, QEMU guest agent. Nie zmieniać działających VM/CT, HA ani FarmAgent. Sprawdzić adres DHCP i rezerwację. Instalować Git i Docker Engine z Compose pluginem z zaufanych źródeł; użyć konta VM, nie hosta Proxmox.

## 2. Przypięty kandydat
Na VM użyć jawnie wskazanego pełnego SHA, uzgodnionego przed uruchomieniem. Nie uruchamiać starego bitcoin_agent.py ani logi. Pobrać tylko pliki programu i konfiguracji:
```sh
mkdir -p "$HOME/gta-test"
cd "$HOME/gta-test"
git init
git fetch --depth=1 --no-tags https://github.com/Siermant/Agent.git PELNY_ZATWIERDZONY_SHA
git rev-parse FETCH_HEAD
git checkout FETCH_HEAD -- global_agent tests ops Dockerfile .dockerignore .gitignore compose.yaml compose.diagnostics.yaml README.md ROADMAP.md
```
Porównać wynik rev-parse z zatwierdzonym SHA przed checkout i uruchomieniem. Nie używać go jako automatycznej akceptacji zmian z GitHub. Narzędzie stage_release.py kwalifikuje obraz odizolowanym testem, lecz nie przełącza działającej usługi. Pierwsza instalacja nie wymaga rollbacku do starej wersji, ale kolejne aktualizacje wymagają kopii i kontrolowanego przełączenia.

## 3. Monitoring
```sh
GTA_PAPER_CRYPTO=0 docker compose up -d --build
docker compose ps
docker compose logs --tail=50 agent
curl -fsS http://127.0.0.1:8080/healthz
docker compose exec -T agent python -m global_agent.preflight
```
Preflight weryfikuje komplet bieżących notowań BTC/ETH, historię zamkniętych świec i wcześniejszy kurs NBP. Kod wyjścia 0 = oba testy zaliczone; 1 = co najmniej jeden błąd. Nie wykonuje zleceń, nie zapisuje historii do bazy. Nie włączać symulacji po błędzie.
Dostęp do panelu przez tunel SSH z komputera w LAN/VPN:
```sh
ssh -N -L 8080:127.0.0.1:8080 UZYTKOWNIK@ADRES_VM
```
Przeglądarka http://127.0.0.1:8080. Nie wystawiać portu do internetu.

## 4. Diagnostyka
Utworzyć osobne prywatne repozytorium Siermant/Agent-diagnostics (z gałęzią domyślną), sprawdzić dostęp połączonego konta GitHub, nadać token tylko do tego repozytorium. Nie udostępniać tokenu w czacie. Włączyć zgodnie z README. Sprawdzić status.json, portfolios.json, daily.json, świeżość dat i brak danych poufnych. Bez repozytorium agent może działać lokalnie.

## 5. Paper trading (tylko po poprawnym preflight)
```sh
GTA_PAPER_CRYPTO=1 docker compose up -d
```
Jeżeli diagnostyka włączona, zachować te same pliki -f compose.yaml -f compose.diagnostics.yaml oraz GTA_DIAG_TOKEN_PATH. Sprawdzić panel, zapis decyzji, restart bez powielonych transakcji, próbę utraty połączenia i kopię zapasową. Nie odłączać całej domowej sieci; test awarii przeprowadzić izolując wyłącznie testowy kontener/VM. Przywrócić połączenie i sprawdzić wznowienie.
Metryki: RAM cgroup v2, limit pamięci, licznik czasu CPU (mikrosekundy od startu kontenera, nie procent), wolne miejsce systemu plików danych. Brak pełnego monitoringu hosta Proxmox. Testy akcji i IKE jeszcze nieaktywne. Nie kupować VPS ani podłączać prywatnego API giełdy.

## Warunki odbioru
- Kontener healthy, 15 portfeli, real_trading=false.
- Aktualne BTC/ETH, poprawny preflight NBP/historii.
- Kopia SQLite i możliwość odczytu raport.zip.
- Diagnostyka prywatna działa, jeśli skonfigurowana.
- Restart bez duplikacji, blokada symulacji przy awarii źródła.
- Żadnego automatycznego wdrożenia develop, rzeczywistych środków ani kluczy brokerskich.
