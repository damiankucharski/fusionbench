# Czy walidacja potrafi wybrać seed wyszukiwania OKAPI?

Na niezależnej serii potwierdzającej wybór najlepszego z 12 drzew według zwykłej walidacji dał średni testowy Brier 0,1783, gdy częstość zmienionego kontekstu wzrosła z 10% do 50%. Losowy seed dał 0,1789. Wybór oparty na częstości kontekstu oszacowanej z 512 nieetykietowanych przykładów docelowych dał 0,1611. Najlepsze dostępne drzewo wybrane po poznaniu etykiet testowych osiągnęło 0,1597. Niższy Brier jest lepszy.

## Co zmieniało się w danych

Siedem symulowanych ekspertów zwraca prawdopodobieństwa binarne. W kontekście referencyjnym każdy jest kalibrowany w swoim marginalnym rozkładzie. Sześciu stabilnych ekspertów dzieli część szumu. W zmienionym kontekście jeden silny ekspert zwraca dopełnienie swojej zwykłej predykcji. Etykieta i predykcje sześciu pozostałych ekspertów są identyczne dla każdej pary kontrfaktycznej.

W każdej z 16 niezależnych prób 12 seedów rzeczywistego wyszukiwania GP otrzymało ten sam zbiór 384 przykładów. Szukano przez 10 pokoleń i wybierano jeden wynik Pareto z najwyższym treningowym wynikiem 1 − Brier na seed. Wszystkie walidacje i testy w próbie porównują ten sam bank 12 zamrożonych drzew. Seria rozwojowa i potwierdzająca mają rozłączne seedy danych i wyszukiwania.

Źródło i zwykła walidacja mają 10% zmienionego kontekstu. Każda próba ma 12 losowań walidacji, z zagnieżdżonymi budżetami 64, 256 i 1024 przykładów. Duża referencja miesza ryzyko dwóch warunków z wagami 90% i 10% na 8192 sparowanych przykładach. Oznaczenie `val_n=16384, repeat=-1` opisuje obliczenie obu warunków na tych 8192 przykładach. Test na osobnym losowaniu 8192 par stosuje wagi odpowiadające częstościom od 0% do 100%. To umożliwia przeprowadzenie interwencji z identycznymi przykładami dla każdego seeda.

## Kiedy ranking się odwraca

| Walidacja | Spearman, test 10% | Spearman, test 50% |
|---|---:|---:|
| 64 przykłady | 0,375 | 0,001 |
| 256 przykładów | 0,688 | −0,043 |
| 1024 przykłady | 0,862 | −0,035 |
| Duża referencja | 0,994 | −0,027 |

Przy budżecie 256 i zmianie 10% → 50% kolejność odwróciło 50,8% porównywalnych par. Średnie tau Kendalla spadło z 0,618 do −0,020. Korelacje używają średnich rang dla remisów, a częstości odwróceń pomijają pary o równych wynikach. Przy potraktowaniu ocen równych do 12 miejsc po przecinku jako remisów odpowiednie korelacje Spearmana wynoszą 0,696 i −0,039.

Dla drzew o ryzykach warunkowych R₀ i R₁ ryzyko po zmianie częstości p jest równe R₀ + p(R₁ − R₀). Przykład z próby 305 wybrano jako największą różnicę na teście 50% wśród par odległych na dużej walidacji o najwyżej 0,001. Drzewo seeda 47 uśredniło trzy wejścia, a seed 49 dodał drugą kopię niestabilnego eksperta do tej samej średniej. Na źródle Brier wyniósł odpowiednio 0,07478 i 0,07422. Przy 50% wyniki wyniosły 0,14730 i 0,19832. Oszacowany na niezależnym teście punkt przecięcia wynosił 10,52%. Ten przykład ilustruje skrajny przypadek z serii.

Praktyczna miara zależności drzewa od kanału to średnia bezwzględna zmiana jego predykcji po odwróceniu wskazanego wejścia. Na osobnej próbie bez użycia etykiet ta miara przewidywała ranking nachylenia R₁ − R₀ ze średnim Spearmanem 0,816. Odpowiednia korelacja dla liczby wystąpień kanału w drzewie wyniosła 0,035, a dla całkowitej liczby węzłów −0,017.

## Wybór przy pomocy próbki z wdrożenia

Metoda wymaga oznaczenia kontekstu na tych samych etykietowanych przykładach walidacyjnych. Szacuje średni wektor 21 niezgodności par ekspertów osobno dla obu kontekstów. Średni wektor z 512 nowych nieetykietowanych przykładów docelowych jest rzutowany na odcinek łączący średnie źródłowe. Wynik określa wagę p w ocenie ryzyka każdego drzewa. W ocenie kontrolnej bez etykiet kontekstu użyto częstości wzorców głosowania ekspertów, z wygładzeniem oszacowań w rzadkich komórkach. Wariant minimaksowy wybiera najmniejsze oszacowane najgorsze ryzyko w z góry zadanym przedziale częstości 10–50%.

| Test | Zwykła walidacja | Oszacowane p | Minimax | Wzorce głosów | Znane p |
|---|---:|---:|---:|---:|---:|
| 10% | 0,00066 | 0,00101 | 0,0057 | 0,0006 | 0,0006 |
| 50% | 0,01854 | 0,00132 | 0,0008 | 0,0158 | 0,0008 |
| 75% | 0,0336 | 0,0006 | 0,0017 | 0,0290 | 0,0006 |

Tabela podaje nadwyżkę testowego Brier ponad najlepsze drzewo z pełnego banku. W próbie potwierdzającej z budżetem 256 przy przejściu do 50% różnica względem zwykłej walidacji wyniosła 0,01722 z 95-procentowym przedziałem t od 0,00702 do 0,02741. Przedział wykorzystuje osiem sparowanych losowań danych, po uśrednieniu powtórzeń walidacji w każdym losowaniu. Przy 10% adaptacja pogorszyła średni wynik o 0,00036. Przy 50% estymator zwrócił średnio 48,1%, ze średnim błędem bezwzględnym 5,68 punktu procentowego. Bez kontekstu w walidacji dany wariant grupowy wraca do wyboru według zwykłej walidacji.

Losowanie czterech spośród 12 seedów i wybór zwycięzcy na walidacji dały przy 50% Brier 0,17876. Dla najgorszej ćwiartki losowań średnia wyniosła 0,18910. Adaptacja poprawiła te wartości do 0,16779 i 0,17791. Obliczenie waży wszystkie równoliczne podzbiory dokładnie i przy remisie wybiera równomiernie.

Generator wymusza zmianę jednego kanału, a częstość tej zmiany pozostawia ślad w niezgodnościach ekspertów. Wartość metody poza tym eksperymentem zależy od stabilności ryzyka wewnątrz grup i możliwości rozpoznania zmiany z nieetykietowanych predykcji. Ten eksperyment nie ustala przyczyny zmian wyników na rzeczywistym zbiorze.

## Odtworzenie

Polecenia zakładają wspólny katalog z checkoutami `fusionbench` i `okapi` z poprawkami oraz interpreter z zależnościami obu projektów.

```bash
export PYTHONPATH="$PWD/fusionbench:$PWD/okapi"
export OKAPI_LOG_LEVEL=ERROR
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m fusionbench.eval.seed_selection --out seed_selection_results/development --data-seeds 100 101 102 103 104 105 106 107
python -m fusionbench.eval.seed_selection --out seed_selection_results/confirmation --data-seeds 300 301 302 303 304 305 306 307 --gp-seeds 40 41 42 43 44 45 46 47 48 49 50 51
```

`config.json` zapisuje wszystkie budżety. `candidates.csv` zawiera drzewa i wyniki interwencji, `validation.csv` pojedyncze oceny walidacyjne, `ranks.csv` porównania rankingów, a `selection.csv` wyniki wyboru z każdego banku. Dalsze tabele pokazują średnie, porównania sparowane i różnice między parami seedów. Regret i najlepszy wynik po fakcie odnoszą się do skończonego banku i skończonej próby testowej.
