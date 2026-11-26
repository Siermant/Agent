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
