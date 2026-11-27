# Pierwsza aktualizacja na VM 106 — v0.10 -> v0.11

Wersja testowa, bez rzeczywistych zleceń. Nie aktualizować z develop w ciemno.
Nie usuwać wolumenu global-trading-agent_agent-data. Nie wykonywać compose down -v.
Zatwierdzenie konkretnego SHA i test na VM są odrębne od kwalifikacji lokalnej.

## Utrwalenie konfiguracji
W istniejącym /opt/global-trading-agent/.env zachować GTA_PAPER_CRYPTO=1.
Dodać lub poprawić (bez powielania kluczy):
```dotenv
COMPOSE_FILE=compose.yaml:compose.diagnostics.yaml
GTA_DIAG_TOKEN_PATH=/etc/gta-secrets/github-token
```
W .env jest jedynie ścieżka do tokenu, nigdy jego treść. Nie commitować .env.
Nie zmieniać zawartości tokenu. Powyższe zakłada już działającą diagnostykę.
Docker Compose nadal używa tego samego projektu i wolumenu.

## Kwalifikacja bez zatrzymywania usługi
W terminalu Debiana, jako root:
```sh
cd /opt/global-trading-agent
mkdir -p /root/gta-qualification
python3 ops/stage_release.py --approved-sha PELNY_SHA_KANDYDATA --output /root/gta-qualification/v011.json
```
Wskazać pełny SHA podany po zapisaniu wydania. Nie używać nazwy gałęzi.
Narzędzie uruchamia testy w izolacji, a nie nową instancję produkcyjną.
Nie uruchamiać ponownie w ciemno po błędzie. Manifest zawiera image_id.
Po kwalifikacji sprawdzić też stan dotychczasowego kontenera i jego wolumenu.
W razie niepowodzenia nie przełączać wersji.

## Kontrolowana aktywacja — osobny etap
Przed aktywacją zachować ID działającego obrazu (docker inspect kontenera), pliki
Compose oraz pełną, sprawdzoną kopię danych poza wolumenem. Przy kopii całego
wolumenu zatrzymać usługę; nie kopiować aktywnego SQLite bez backup API.
Nie udostępniać treści tokenu ani nie kopiować go do repozytorium.
Wersja 0.11 zmienia panel i trasowanie, bez migracji schematu SQLite.
Nowy obraz uruchomić z tą samą konfiguracją diagnostyki i dotychczasowym
wolumenem, dopiero po zatwierdzeniu i wykonaniu kopii. Zweryfikować healthz,
wersję API, liczbę 15 portfeli, wcześniejsze wykonania, real_trading=false,
stan paper i świeżość raportów. Nie uruchamiać równocześnie dwóch wykonawców.
Zachować stary obraz; rollback w tym wydaniu może użyć starego obrazu z
niezmienionym schematem danych, bez kasowania transakcji. Przy niespójności
zatrzymać usługę i zweryfikować kopię, nie nadpisywać automatycznie bazy.
Automatyczne przełączenie i rollback nadal nie są zaimplementowane.
