# Global Trading Agent

**Wersja rozwojowa 0.10.0-dev — monitoring i opcjonalny paper trading BTC/ETH.** Nie wykonuje rzeczywistych zleceń ani nie posiada integracji z prywatnym API brokera. Nie uruchamiać starego bitcoin_agent.py. Stary kod i logi pozostają archiwalne; nie należy ich kopiować do obrazu ani używać ujawnionych kluczy.

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
Backtest jest osobnym trybem badawczym; nie modyfikuje portfeli symulacji bieżącej.
Przykład pobrania historii w istniejącym kontenerze (po weryfikacji połączenia i warunków API):
```sh
docker compose exec agent python -m global_agent.history --start 2025-01-01 --end 2025-12-31 --symbols BTC-EUR ETH-EUR --output /data/crypto_2025.csv
docker compose exec agent python -m global_agent.backtest /data/crypto_2025.csv --kind crypto --benchmark BTC-EUR
```
Źródła: OKX EEA history-candles, 1Dutc, tylko confirm=1; NBP tabela A EUR, zawsze kurs opublikowany przed dniem świecy. Kurs referencyjny nie jest ceną rzeczywistej wymiany waluty. Brakujące dni powodują błąd, bez interpolacji. Zakres 21–730 pełnych dni; dostępność par i historii zależy od API. Nie uruchamiać wielu pobrań równocześnie.
Alternatywnie dostarczyć CSV date,symbol,open_pln,close_pln, z jednolitym, sprawdzonym kalendarzem sesji i cenami w PLN. Pierwsze 20 sesji stanowi rozgrzewkę, sygnał z poprzednich zamknięć, wykonanie na otwarciu następnej sesji. Każde uruchomienie testuje 5 niezależnych wariantów dla wybranego rodzaju rachunku; nie modyfikuje bazy monitoringu. Wyniki JSON na stdout. Koszty domyślne to założenia: prowizja 0,25%, pełny spread 0,2%, poślizg 0,1% na stronę; parametry konfigurowalne. To nie aktualny cennik giełdy. Akcje domyślnie w całych sztukach, krypto z założonym krokiem 0,000001. D jest regułowy, nie AI. C nie jest jeszcze pełnym gridem. Brak podatków, dywidend, corporate actions, intraday, danych o płynności i rzeczywistych kalendarzy akcji. Testować małe uniwersum; aktywne portfele i benchmark mają różne ekspozycje. Nie interpretować wyniku jako prognozy.

## Paper trading v0.6 — pierwsza automatyczna symulacja
Po sprawdzeniu aktualnych notowań, historii i FX na VM, uruchomić:
```sh
GTA_PAPER_CRYPTO=1 docker compose up -d --build
docker compose logs --tail=50 agent
```
Wyłączenie: `GTA_PAPER_CRYPTO=0 docker compose up -d`. Bez parametru domyślnie monitoring; brak realnych zleceń w obu trybach. Zachować ten sam wolumen. Decyzje raz na dobę UTC dla BTC-EUR i ETH-EUR, z 20 zamkniętych świec dziennych, realizacja po aktualnym bid/ask (nie po historycznym otwarciu). Kurs NBP z poprzedniego dnia publikacji, referencyjny dla PLN. Założenia kosztów: 0,25% prowizji, 0,1% poślizgu, krok ilości 0,000001, bez modelu głębokości rynku. Nie są aktualną taryfą OKX ani ceną wykonalną FX. Warianty A–D: pozycje do 5%, budżet zakupu 4,5%; tylko dwa instrumenty, zatem aktywna ekspozycja początkowo około 9%, a benchmark może mieć niemal 100%. Nie porównywać zysku bez uwzględnienia różnic ryzyka. E kupuje BTC jednokrotnie. Pozostałe 10 portfeli (akcje i IKE) nie handluje.
Brak świeżego kompletu cen, historii lub kursu blokuje cykl; nie odtwarzamy utraconych transakcji po fikcyjnych cenach. Rejestr paper_decisions i deterministyczne ID zleceń zapobiegają powtórzeniu wykonania w tej samej dobie; lokalny flock chroni przed równoległym cyklem na tym samym dysku. Bez automatycznego failover. Nie jest to scalping ani pełny grid, D nie jest jeszcze AI. Brak podatków, zleceń ochronnych i modelu częściowych realizacji. Wersja rozwojowa: najpierw smoke test w środowisku docelowym. Panel przez SSH/VPN, bez publicznego portu.

## Diagnostyka i kopie v0.7
Panel pokazuje zwrot od początku, bieżące obsunięcie od szczytu (nie historyczne maksimum), prowizje i status aktualności wyceny. API status zawiera też ostatnią zapisaną wycenę na każdy dzień z 30 dni UTC, bez uzupełniania luk. Brak wyceny nie oznacza zerowego wyniku. Porównanie krypto A–E nie wyrównuje ekspozycji ani podatków.
Przycisk Pobierz raport.zip eksportuje tylko wybrane informacje symulacyjne w JSON: status, statystyki portfeli, liczbę decyzji według typu. Nie eksportuje surowych logów, baz, środowiska ani kluczy. Raport nadal zawiera dane symulowanych portfeli, nie publikować go automatycznie w repozytorium. Panel pozostaje dostępny przez tunel SSH/VPN, bez publicznego portu.
Proces co 5 minut sprawdza potrzebę lokalnej kopii dziennej UTC, zachowuje 7 ukończonych kopii pod /data/backups. Kopie powstają przez SQLite backup API, są sprawdzane quick_check, publikowane po ukończeniu. Bazy są spójne indywidualnie, nie stanowią atomowej pary. Kopie są na tym samym wolumenie, nie zastępują kopii Proxmox/NAS. Przy odtwarzaniu zatrzymać usługę i odtworzyć kopie do osobnego katalogu, zweryfikować stan; nie nadpisywać czynnej bazy i nie uruchamiać równolegle drugiego wykonawcy. Brak automatycznego odtwarzania i zdalnej synchronizacji.

## Diagnostyka zdalna v0.8 (opcjonalna, domyślnie wyłączona)
Docelowe oddzielne repozytorium: Siermant/Agent-diagnostics, prywatne, nie fork, z zainicjalizowaną gałęzią domyślną. Nie zmieniać jego widoczności na publiczną — historia Git zachowuje wcześniejsze dane. Wysyłka sprawdza prywatność przed każdym zapisem; nie eliminuje to ryzyka zmiany widoczności przez administratora. Użytkownik tworzy repozytorium i nadaje poświadczenia dopiero podczas instalacji. Fine-grained PAT wyłącznie dla repozytorium diagnostycznego, Contents read/write, Metadata read, bez uprawnień administracyjnych, workflow i do repozytorium kodu. Token przechowywać poza repozytorium w pliku dostępnym do odczytu przez UID 10001, bez dostępu innych użytkowników. Nie przesyłać tokenu w rozmowie ani logach.
Uruchomienie po przygotowaniu repozytorium i tokenu:
```sh
GTA_DIAG_TOKEN_PATH=/sciezka/poza/repo/token docker compose -f compose.yaml -f compose.diagnostics.yaml up -d --build
```
Jeśli symulacja ma być włączona, zachować ustawienie GTA_PAPER_CRYPTO=1 zgodnie z wcześniejszą instrukcją. Bez pliku override wysyłka wyłączona. Raporty status.json co 15 minut i przy wykrytej zmianie stanu, portfolios.json co godzinę, daily.json raz dziennie UTC; kontrola co minutę przy sprawnej komunikacji. Raporty zawierają dane symulacji, nie surowe logi, hasła, bazy, adresy IP ani klucze. Kolejka przechowuje najnowszy oczekujący raport każdego rodzaju, nie pełną historię zdarzeń; ponowienia z narastającą przerwą do 15 minut. Każda aktualizacja pliku zostawia historię Git. Brak możliwości natychmiastowego raportowania przy awarii VM lub internetu; ostatni timestamp sygnalizuje potencjalną przerwę. Brak zdalnego wykonywania poleceń, zdalnych aktualizacji i metryk CPU/RAM hosta. Nie wystawiać SSH ani panelu publicznie. Weryfikacja rzeczywistej wysyłki wymaga VM i repozytorium; dotychczas testy z atrapą API.

## Kwalifikacja wydania (etap v0.9)
Na VM, z zaufanej lokalnej kopii narzędzia ops/stage_release.py i z uprawnieniami do Docker, podać pełny, zatwierdzony SHA konkretnego commitu zawierającego narzędzie oraz testy. Nie podawać develop, main ani automatycznie pobranego SHA. Nie wykonywać skryptu pobranego z kanału diagnostyki.
```sh
python3 ops/stage_release.py --approved-sha PELNY_ZATWIERDZONY_SHA --output "$HOME/gta-qualification/qualified-NOWA-WERSJA.json"
```
Katalog manifestu ma być poza repozytorium, plik nie może wcześniej istnieć. Narzędzie pobiera wskazany commit, wybiera tylko katalogi programu, testów, ops i pliki budowy, buduje obraz z tagiem kandydata i zapisuje jego niezmienny lokalny ID. Testy jednostkowe i próba HTTP działają w odizolowanych kontenerach bez sieci, poświadczeń, portów publicznych i właściwego wolumenu danych. Nie zmienia działającej usługi Compose. Manifest powstaje dopiero po poprawnym przejściu testów. Nie jest podpisem kryptograficznym ani zgodą na instalację. Obraz bazowy nadal nie jest przypięty do digestu; identyczny commit nie gwarantuje odtwarzalnej budowy.
Przed aktywacją: potwierdzić wynik testu rzeczywistych źródeł danych na VM, zanotować ID starego obrazu i konfigurację Compose, zatrzymać wykonawcę, wykonać zweryfikowaną kopię wolumenu poza nim oraz sprawdzić odtwarzanie. Do uruchomienia nowej wersji użyć osobnego sklonowanego wolumenu; zachować poprzedni obraz i oryginalny wolumen. Nie uruchamiać dwóch wykonawców jednocześnie. W razie niepowodzenia zatrzymać nową instancję i wrócić do poprzedniego obrazu oraz jego wolumenu; zachować nowy wolumen do uzgodnienia symulowanych decyzji, nie wykonywać automatycznego nadpisania danych. Powyższa procedura aktywacji i rollbacku jest jeszcze ręczna i niezweryfikowana na Dockerze. Skrypt kwalifikacji nie wykonuje kopii działającego wolumenu, nie przełącza wersji i nie stanowi aktualizatora produkcyjnego.

## Kandydat v0.10 — niezawodność i odbiór
- Blokada flock całego procesu dla jednego lokalnego katalogu danych. Nie zapewnia blokady rozproszonej pomiędzy różnymi VM lub wolumenami.
- Po dziennej kopii baza wycen zachowuje wszystkie punkty z ostatnich 7 dat UTC i ostatni zapis na dzień dla starszych dat. Nie usuwa wykonań ani decyzji; zachowuje szczyt i podstawy limitów ryzyka. Zwolnione strony SQLite mogą być ponownie użyte, plik nie musi fizycznie maleć. Starszej historii intraday nie należy traktować jako kompletnej.
- Panel pokazuje do 50 ostatnich wykonań; API status, raport.zip (executions.json) i prywatny daily.json do 200 ostatnich symulowanych wykonań znanych symboli, bez identyfikatorów zleceń. Pełna księga pozostaje lokalnie. Dla diagnostyki zdalnej nadal wymagana prywatność repozytorium i zgoda na wysyłkę danych.
- Instrukcja startu: ops/FIRST_BOOT.md. Najpierw monitoring, następnie preflight i dopiero po udanym teście rzeczywistych źródeł paper trading. Nie uruchamiać starego agenta. Wcześniej ujawnione poświadczenia należy unieważnić.
- Testy: python3 -m unittest discover -s tests -v; obejmują także rzeczywiste HTTP na loopback bez internetu. Docker, OKX/NBP i zdalna diagnostyka wymagają weryfikacji na VM. Automatyczna aktywacja aktualizacji i rollback niegotowe.
