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
