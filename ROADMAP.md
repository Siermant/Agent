# Global Trading Agent — plan projektu
Stan: 2026-11-26. Gałąź develop; istniejący bitcoin_agent.py pozostaje bez zmian.

## Uzgodnione
- Praca 24/7 na Linux/Proxmox; rozwój bezpośrednio przez GitHub po konsultacji.
- Najpierw wyłącznie paper trading, backtesting i demo, bez prawdziwych zleceń.
- Globalne akcje, ETF-y, kryptowaluty; surowce przez dostępne instrumenty bez dźwigni. Forex i pochodne odłożone.
- Niezależne portfele, wspólne dane i skaner; porównanie strategii po prowizjach, spreadzie i poślizgu.
- Kandydaci: IBKR (akcje), OKX/Revolut X (krypto); Bossa IKE tylko po pisemnym potwierdzeniu zakresu API. Nie obchodzimy ograniczeń dostępu.
- IKE symulowane osobno; polska ewidencja podatkowa i NBP, osobne zasady krypto, akcji i dywidend. Nie zakładamy dostępności IKE przez API.
- Panel WWW, logi diagnostyczne bez sekretów, eksport raport.zip, wersjonowanie, aktualizacja wyłącznie zatwierdzonego wydania z rollbackiem.
- Żadnych kluczy API, danych finansowych ani prywatnych logów w repozytorium. Wcześniej ujawnione klucze należy unieważnić.

## Parametry startowe (propozycje konfigurowalne)
- Kapitał wirtualny: IKE 28 260 PLN, globalny 50 000 PLN, krypto 20 000 PLN; osobne portfele, bez transferów między nimi.
- Maks. pozycja 5% wartości portfela, minimalna gotówka 20%, limit dzienny 2%, tygodniowy 5%, drawdown 10%; limity nie gwarantują maksymalnej straty.
- Brak shortów, margin i dźwigni; blokada nowych pozycji przy awarii lub przekroczeniu limitów, możliwość redukcji ryzyka.
- W IKE osobne limity i instrumenty dopuszczone do obrotu; kwota symulacyjna nie jest rzeczywistą wpłatą.

## Etapy
1. Trwała księga paper trading, walidacja zleceń, idempotencja, testy jednostkowe.
2. Dane historyczne i live: identyfikatory instrumentów, kursy walut, sesje, świeżość danych, koszty.
3. Backtest bez look-ahead, walk-forward i porównanie z benchmarkiem; strategie trend, mean reversion, breakout, grid i momentum.
4. Skaner rynków, panel WWW i raporty; moduł podatkowy z uzgodnieniem raportów.
5. Długotrwała symulacja (proponowane 60–90 dni), testy awarii, demo brokerów.
6. Ewentualny handel realny tylko po osobnej decyzji użytkownika, potwierdzeniu API i walidacji. Nie jest częścią pierwszego wydania.

## Następny krok
Rozwinąć i uruchomić silnik symulacji. Nie wdrażać automatycznie na main ani na Proxmox. Zweryfikować testy przed instalacją.

## Porównanie wariantów (zatwierdzone)
- A: trend, profil zachowawczy; cel ekspozycji 40%.
- B: trend, mean reversion, breakout; cel 60%.
- C: breakout, momentum, dynamic grid; cel 80%.
- D: adaptacyjny wybór strategii, docelowo AI; cel do 80%.
- E: pasywny benchmark: globalny ETF UCITS dla IKE i global, BTC spot dla krypto.
- Cztery aktywne warianty po trzy niezależne portfele oraz trzy benchmarki, razem 15 portfeli o identycznych kapitałach w obrębie rodzaju rachunku. Łączny nominalny kapitał symulacji 491 300 PLN, nie kapitał rzeczywisty.
- Wspólny czas startu, dane dostępne w danym momencie, koszty właściwe dla brokera, dywidendy, FX, realistyczne wykonanie. Raportujemy także wpływ odmiennej ekspozycji i ryzyka.
- Metryki: stopa zwrotu netto, drawdown, Sharpe, profit factor, liczba transakcji, prowizje, błędy i wynik względem benchmarku. Cele badawcze 10–15% rocznie nie są prognozą ani warunkiem wymuszającym handel.
- Moduł experiments.py definiuje profile i izolowane portfele w pamięci; strategie nie są jeszcze wykonywane, benchmark nie jest jeszcze wyceniany. Kolejny krok: trwała księga i testy.

## Architektura hybrydowa (zatwierdzona)
- Pierwsze uruchomienie paper trading na osobnej VM Ubuntu Server 24.04 LTS w Proxmox, początkowo 2 vCPU i 3–4 GB RAM, po weryfikacji wolnych zasobów. Bez GUI; panel WWW.
- Jedna aplikacja i wspólne dane dla 15 portfeli, lekkie modele ML, zadania treningowe z limitami CPU i RAM. SQLite WAL z jednym procesem zapisującym, dane historyczne Parquet w miarę potrzeb.
- Przenośna instalacja Docker Compose, identyczna lokalnie i w chmurze; trwałe wolumeny, healthcheck, restart, wersjonowane migracje i kopie zapasowe.
- Docelowo VPS w UE jako niezależna instancja główna: dane live, silnik strategii, kontrola ryzyka, księga, panel. Kandydat OVH VPS-2 (4 vCPU, 8 GB RAM); cenę i warunki sprawdzić przed zamówieniem. Bez zamówienia na tym etapie.
- Proxmox: backtesting, trening, testy nowych wersji, szyfrowane kopie zapasowe i środowisko odtworzeniowe. Połączenie WireGuard; awaria domu nie może blokować głównej instancji VPS.
- Nie synchronizować aktywnego pliku SQLite przez sieć; wykonywać spójne kopie przez SQLite backup API i testować odtwarzanie.
- Tylko jedna instancja wykonawcza. Przejęcie awaryjne początkowo ręczne po potwierdzeniu zatrzymania poprzedniej instancji i uzgodnieniu zleceń oraz pozycji. Brak automatycznego failover przy ryzyku split-brain.
- Blokować nowe pozycje przy nieaktualnych danych, utracie połączenia, niepewnym stanie zlecenia, niespójności salda lub awarii kontroli ryzyka. Docelowo zlecenia ochronne po stronie brokera, jeśli obsługiwane. Brak gwarancji ceny wykonania.
- Panel dostępny przez VPN, uwierzytelnianie, monitoring heartbeat z niezależnego miejsca, alerty awarii, klucze poza repozytorium, bez uprawnień do wypłat. IB Gateway: uwzględnić ręczne uwierzytelnianie i wygaśnięcie sesji.
- Żadnego automatycznego wdrożenia gałęzi develop; aktualizacje z zatwierdzonych wydań, test przed przełączeniem i procedura rollback uwzględniająca bazę danych.
- Nadal tylko symulacja; rzeczywiste transakcje wymagają odrębnej decyzji i walidacji. VPS i VM nie zostały jeszcze utworzone.

## Decyzja wdrożeniowa 2026-11-26
- Pierwsza wersja testowa wyłącznie na Proxmox; zakup VPS odłożony do późniejszej decyzji. Zachować przenośność Docker Compose i docelową architekturę hybrydową.
- Przerwa w dostępie do internetu: wstrzymać symulowane wykonania bez świeżych danych, zapisać awarię, po powrocie połączenia zweryfikować dane i stan portfeli; nie symulować realizacji po nieznanych cenach.
- Parametry VM: 2 vCPU host, 3 GB RAM, 40 GB SSD, Ubuntu Server 24.04 LTS, po sprawdzeniu zasobów hosta. Panel WWW po polsku przez LAN/VPN.

## Zasady współpracy (zatwierdzone)
- Samodzielnie realizować zwykłe decyzje techniczne, implementacje, testy, poprawki i dokumentację na develop; nie zatrzymywać prac z prośbą o potwierdzenie każdego kroku.
- Osobna zgoda użytkownika: rzeczywiste środki lub zlecenia, podłączenie uprawnień transakcyjnych, płatne usługi, istotna zmiana limitów ryzyka, publikacja danych prywatnych, destrukcyjne operacje i wdrożenie produkcyjne.
- Praca nad kodem odbywa się podczas aktywnych sesji; po wdrożeniu proces agenta będzie niezależnie działać całodobowo.

## Stan implementacji v0.4.0-dev
- Dodano global_agent/__main__.py: jednowątkowa pętla odczytu OKX co 30 s, panel tylko do odczytu, API status i healthcheck; brak automatycznych strategii i rzeczywistych zleceń.
- Dockerfile, compose.yaml, .dockerignore, .gitignore, README oraz workflow testów. Kontener bez roota, limit 1 CPU i 768 MB RAM, trwały wolumen; panel tylko loopback VM, dostęp tunelem SSH. Nie kopiujemy starego kodu ani logów do obrazu.
- Lokalnie 10/10 testów jednostkowych (Python 3.13) na danych syntetycznych oraz 4/4 prób HTTP. Porównano SHA blobów pięciu testowanych plików z GitHubem, identyczne. GitHub Actions skonfigurowano dla Python 3.12, lecz podczas sprawdzenia brak uruchomień. Nie zweryfikowano budowy obrazu ani komunikacji z OKX na żywo (środowisko bez DNS zewnętrznego i Docker). Dokumentacja OKX potwierdza domenę eea.okx.com i publiczny endpoint tickers.
- Następne prace bez dodatkowej akceptacji: historyczne zamknięte świece, FX i zasady świeżości, silnik backtest, pierwsze strategie i benchmark, testy integracyjne z symulowaną awarią, rozszerzenie danych akcji. Nie włączać transakcji realnych.

## Stan v0.5 — badania historyczne (2026-11-26)
- Dodano backtest.py: odseparowana tymczasowa księga, 5 wariantów A–E dla wybranego rodzaju rachunku; 20 zamkniętych sesji rozgrzewki, sygnały z poprzednich zamknięć, wykonania po następnym otwarciu. Wycenia w PLN z parametrycznym kosztem spreadu, poślizgu i prowizji. Raport JSON: zwrot, obsunięcie na zamknięciach, liczba wykonań, blokady i seria kapitału.
- D jest regułowy, C nie jest jeszcze prawdziwym Dynamic Grid; nie ma modelu AI, testów walk-forward ani realistycznych danych płynności, dywidend i podatków. Benchmark ma inną ekspozycję niż aktywne warianty. Koszty przykładowe, nie taryfa brokera.
- Dodano history.py: publiczna historia dziennych świec OKX EEA EUR spot (1Dutc, confirm=1, paginacja), kurs NBP EUR z daty ściśle wcześniejszej, CSV z pochodzeniem i kursem. Braki dni są błędem, bez dopisywania sztucznych danych. Kurs NBP nie jest ceną wykonalną FX.
- Lokalnie 18/18 testów unittest na danych syntetycznych i atrapach API. SHA nowych czterech plików na GitHubie zgodne z lokalnymi plikami testowanymi. Brak DNS zewnętrznego w środowisku, nie wykonano prawdziwego pobrania historii. Nie wdrożono na Proxmox. Panel nadal 0.4 monitor-only; v0.5 to osobny tryb badawczy CLI.
- Następne: rzeczywisty smoke test na VM, zoptymalizowany backtest, raport porównawczy, źródło akcji i sesje oraz mechanizm paper trading na bieżących notowaniach; nadal brak realnych zleceń.

## Stan v0.6.0-dev — automatyczna symulacja krypto
- paper_live.py: BTC-EUR i ETH-EUR; 20 zakończonych świec 1Dutc, decyzja raz na dobę UTC, aktualny bid/ask oraz referencyjny wcześniejszy EUR/PLN NBP. Warianty A–D regułowe i E benchmark BTC; 5 portfeli aktywnych symulacyjnie po włączeniu flagi, 10 portfeli akcji/IKE nieaktywnych. Ze względu na małe uniwersum i limit pozycji 5% aktywne warianty mają znacznie niższą ekspozycję niż benchmark. Koszty testowe 0,25% i 0,1% slippage, nie taryfa brokera.
- Dziennik decyzji w SQLite, deterministyczne ID, odzyskanie po przerwie między zapisem wykonania i decyzji, blokada lokalna flock. Brakujące/stare notowania lub FX, niekompletna historia i awaria blokują cykl; bez retroaktywnych wykonań. Brak automatycznego handlu realnego.
- Integracja z pętlą i panelem; GTA_PAPER_CRYPTO=1 włącza symulację, domyślnie 0 do smoke testu na VM. Zaktualizowano compose i README.
- Lokalnie wykonano 24/24 testów (18 wcześniejszych i 6 nowych); trzy pliki kodu/testów identyczne z blobami na GitHubie. Testy syntetyczne, nie dowód rentowności. Nie zweryfikowano połączenia OKX/NBP ani budowy kontenera w środowisku docelowym, brak wdrożenia. Następne: diagnostyka, raport i backup, szersze uniwersum i dane tradycyjnych rynków, kalendarze i FX, testy walk-forward i symulacja intraday. Nadal paper-only.

## Stan v0.7.0-dev — diagnostyka i lokalne kopie
- diagnostics.py: statystyki portfeli (zwrot od startu, bieżące obsunięcie, prowizje, obroty, liczba wykonań i ostatnia wycena każdego z zapisanych dni z 30 dni UTC), status świeżości. Brak wyceny pozostaje null. Porównanie A–E w panelu dla krypto; brak podatków i wyrównania ekspozycji.
- raport.zip z panelu: trzy pliki JSON z dozwolonymi danymi symulacyjnymi, bez surowych baz, logów i kluczy. Nadal dane prywatne, nie publikować na GitHubie.
- backup.py: dzienna lokalna kopia SQLite obu baz, quick_check, 7 ukończonych kopii, status w panelu, niezależny wątek sprawdzający co 5 minut. Bazy spójne indywidualnie, nie atomowa para. Ten sam wolumen — brak odporności na utratę dysku i brak kopii offsite. Odtwarzanie ręczne po zatrzymaniu usługi.
- Lokalnie 29/29 unittest i 4/4 prób HTTP (w tym poprawny ZIP), cztery zmienione/nowe pliki mają identyczne SHA z GitHubem. Nadal brak smoke testu Docker/OKX/NBP na VM i danych z rzeczywistego rynku. Tryb paper krypto domyślnie wyłączony, real trading niedostępny.
- Następne: dane akcji i ETF, kalendarze sesji, rozszerzenie uniwersum, intraday i walk-forward; test instalacji na Proxmoxie oraz kopia poza wolumenem. Nie wdrażać produkcyjnie.

## v0.8 — komunikacja diagnostyczna
- Oddzielny prywatny cel Siermant/Agent-diagnostics, ograniczony token Contents read/write wyłącznie do tego repozytorium, poświadczenie poza repo; domyślnie wyłączone. Nie tworzono repozytorium diagnostycznego ani tokenu. Włączenie dopiero po przygotowaniu ich na VM. Nigdy nie publikować repozytorium raportów.
- remote_diag.py i compose.diagnostics.yaml: status co 15 minut i przy zmianie stanu, statystyki co godzinę, dzienne dane zbiorcze; kolejka trzech najnowszych oczekujących raportów i ponowienia z backoff. Weryfikacja prywatności przed zapisem, zakaz przekierowań HTTP. Brak pełnej historii awarii, metryk hosta i surowych logów. Brak kanału poleceń.
- 33/33 unittest lokalnie, trzy pliki zgodne SHA z GitHubem. Mock API, brak prawdziwej transmisji i testu Compose na VM.
- Oddzielić mechanizm aktualizacji od diagnostyki. Nie uruchamiać zdalnych komend ani kodu z repozytorium raportów. Docelowo zatwierdzone wydanie z przypiętym SHA/digestem, testy, kopia i weryfikacja migracji, kontrolowany restart i rollback. Automatyczny updater nie jest jeszcze zaimplementowany. Nie wdrażać develop automatycznie.

## Etap v0.9 — kwalifikacja wydania przed instalacją
- ops/stage_release.py: ręcznie wskazany pełny SHA zatwierdzonego commitu, weryfikacja pobranego SHA, budowa lokalnego obrazu kandydata i identyfikacja po image ID, testy jednostkowe oraz smoke test HTTP w izolacji bez sieci, tokenów i danych działającej usługi. Manifest kwalifikacji tylko po sukcesie. Brak automatycznego wdrażania develop, brak poleceń z repozytorium diagnostycznego.
- tests/test_stage_release.py: 4/4 nowe testy lokalne z symulacją poleceń Git/Docker; weryfikacja SHA, izolacja, odrzucenie kandydata po błędzie testów i czyszczenie kontenera testowego. Oba pliki zgodne SHA z GitHubem. Poprzedni udokumentowany zestaw 33 testów nie został w tej sesji ponownie uruchomiony. Nie wykonano rzeczywistej budowy Docker (brak Docker w środowisku).
- Aktywacja, kopia całego wolumenu poza nim, przełączenie na sklonowany wolumen i rollback pozostają ręcznym planem, nie zaimplementowano automatycznego aktualizatora. Przed automatyzacją przeprowadzić test instalacji i odtworzenia na VM; zachować poprzednią wersję i obie kopie danych. Panel aplikacji nadal raportuje 0.8.0-dev; etap v0.9 dotyczy narzędzi wdrożeniowych.
- Następne prace: kontrolowane wdrożenie i rollback z zachowaniem danych, metryki zasobów, smoke test OKX/NBP i prywatnej diagnostyki, następnie akcje/ETF oraz szersze badania strategii. Real trading niedostępny.
