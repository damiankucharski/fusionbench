# Czy pojedyncza sieć uczy się rozpoznawać zmianę wiarygodności czujnika?

W symulacji o zmiennej polaryzacji drugiego czujnika dostęp do 32 wcześniejszych, nieetykietowanych obserwacji obniżył testowy Brier jednego klasyfikatora z 0,1730 do 0,1357. Różnica wyniosła −0,0373 z 95-procentowym przedziałem od −0,0393 do −0,0352. Oddzielnie trenowany obserwator i głowica o tej samej architekturze uzyskały 0,1356. Eksperyment wskazuje, jaka informacja umożliwia adaptację, i pokazuje ograniczenie korzyści ze wspólnego treningu.

## Dane i informacja dostępna modelom

Każdy przykład ma zbalansowaną etykietę Y ∈ {−1, +1}. Pierwszy czujnik mierzy 0,65Y z niezależnym szumem normalnym o odchyleniu 1. Drugi mierzy 1,5Y z takim samym szumem i odwraca znak sygnału z prawdopodobieństwem p. W obu kanałach marginalny rozkład surowego pomiaru jest stały względem p. Ich zależność dostarcza informacji o p, ponieważ E[UV | p] = 0,975(1 − 2p).

Sieć uczy się na surowych pomiarach i prawdziwych etykietach. Koduje każdy element zbioru kontekstowego, uśrednia reprezentacje i wyprowadza liczbę z zakresu [0,1]. Głowica otrzymuje dwa surowe pomiary badanego przykładu, oszacowany kontekst oraz iloczyn drugiego pomiaru i oszacowanej orientacji 1 − 2p. Ta ostatnia cecha jest zadanym założeniem architektury. Sieć ma 278 parametrów. Wspólną reprezentację kontekstu oblicza się raz na partię. Każdy porównywany klasyfikator zaczyna od losowych wag i uczy się bez predykcji eksperta.

Jeden trening obejmuje 320 aktualizacji, każda na 24 nowych partiach po 32 przykłady kontekstowe i 16 osobnych przykładów z etykietami. Trening monitora, jeżeli jest używany, dostaje nominalne p partii ustalone w generatorze. W źródle p przyjmuje wartości 0, ¼, ½, ¾ lub 1. Test zawiera po 40 partii dla każdego p z zestawu 0,05, 0,25, 0,50, 0,75 i 0,95. Na każdą partię przypadają 64 nowe przykłady z ukrytymi etykietami. Dwie próby pilotażowe i osiem potwierdzających mają rozłączne seedy. Liczby poniżej są średnimi po ośmiu próbach potwierdzających. Przedziały t dla różnic używają sparowanych seedów.

## Który element jest potrzebny

| Wariant | Brier | Trafność |
|---|---:|---:|
| Głowica statyczna | 0,1730 | 74,21% |
| Obserwator trenowany z etykietą p, głowica trenowana z Y | 0,1356 | 80,68% |
| Wspólny trening na p i Y | 0,1357 | 80,68% |
| Trening wyłącznie na Y | 0,1362 | 80,58% |
| Ta sama głowica ze znanym p | 0,1227 | 82,58% |
| Dokładny predyktor Bayesa ze znanymi p i parametrami generatora | 0,1200 | 82,83% |

Obserwator trenowany wyłącznie z etykietą p ma te same warstwy, inicjalizację i dane co sieć trenowana wspólnie. Zatrzymanie gradientu z błędu zadania przy skalarze p dało różnicę Brier poniżej 0,0001 ze sparowanym przedziałem [−0,00028, 0,00038] dla treningu wspólnego względem rozdzielonego.

Podmieniliśmy wyłącznie zbiór kontekstowy na równoliczny zbiór z częstością 1 − p, zachowując te same klasyfikowane przykłady. Brier sieci trenowanej wspólnie wzrósł o 0,1973 względem podmiany na świeży zbiór z właściwym p. W kontroli negatywnej dwa czujniki w każdym przykładzie kontekstowym mierzyły dwa niezależne zjawiska. Ich łączny rozkład był wtedy identyczny dla każdego p. Oddzielnie wytrenowana sieć z tym kontekstem uzyskała 0,1729, a różnica względem klasyfikatora statycznego wyniosła −0,00007 [−0,00020, 0,00005]. Właściwe przykłady klasyfikacyjne nadal pochodziły z informacyjnej pary czujników.

## Czy skalar nauczony tylko z Y daje się zinterpretować

Sam błąd klasyfikacji pozwala na odwrócenie orientacji wewnętrznej skali. Po treningu zamroziliśmy sieć i dopasowaliśmy do jej skalaru odczyt liniowy z wyrazem wolnym na oddzielnych partiach źródłowych z oznaczonym p. Ograniczyliśmy odczyt do [0,1]. W pięciu z ośmiu prób współczynnik kierunkowy był ujemny. Bez kalibracji RMSE surowego wyjścia względem p wynosił 0,3713. Po 8 oznaczonych partiach wyniósł 0,1763, po 128 wyniósł 0,1531. Pełny trening z etykietą p dał 0,1516. Kalibracja dotyczy tylko raportowanego p i zachowuje wszystkie decyzje klasyfikatora. Wynik zależy od jednowymiarowego kontekstu i obecności skalarnego ograniczenia w architekturze.

## Zmiana wzmocnienia i granica zastosowania

Po pomnożeniu drugiego kanału przez 0,25 albo 4, surowa sieć trenowana wspólnie uzyskała Brier odpowiednio 0,1581 i 0,1784. Osobna sieć, która w treningu i wdrożeniu dzieliła oba kanały przez ich RMS z partii kontekstowej, uzyskała 0,1369 przy każdym z trzech wzmocnień. Dla dodatniego mnożnika stałego w obrębie partii dzielenie algebraicznie usuwa zmianę skali. Przy oryginalnej skali koszt tego założenia wyniósł +0,0012 Brier względem sieci bez normalizacji. Zmiana prawa szumu, jakości pierwszego czujnika albo różnych skal w zbiorach kontekstowym i klasyfikacyjnym wymaga osobnej próby.

Eksperyment dotyczy monitorowania wiarygodności na danych syntetycznych. Pomiary referencyjne ze znanym p dostają informację uprzywilejowaną. Kontrola Bayesowska dodatkowo zna dokładne parametry szumu i wzmocnienia. Otwarta hipoteza dotyczy tego, czy dwa jednoczesne rodzaje zmian można odróżnić ze wspólnej reprezentacji i jak często monitor powinien zasygnalizować niepewność.

## Odtworzenie

W katalogu głównym repozytorium, w środowisku z PyTorch i zależnościami FusionBench:

```bash
python -m fusionbench.eval.context_student --output docs/ensemble-study/context-student/pilot --seeds 0 1
python -m fusionbench.eval.context_student --output docs/ensemble-study/context-student/confirmation --seeds 200 201 202 203 204 205 206 207
python docs/ensemble-study/context-student/summarize.py
python -m pytest -q tests/test_context_student.py
```

Katalogi serii zawierają konfigurację, miary dla każdego seeda, wyniki według p, współczynniki kalibracji, średnie i sparowane przedziały. `context_rmse` dla kontrolnych głowic statycznej i ze znanym p opisuje pomocniczy odczyt monitora. Głowice te w predykcji używają odpowiednio stałego i prawdziwego kontekstu. W Brier binarnym użyto pojedynczej kolumny prawdopodobieństwa zdarzenia.
