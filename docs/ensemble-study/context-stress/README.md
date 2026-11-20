# Kiedy obserwator zmiany kontekstu traci wiarygodność?

Poprzedni eksperyment pokazał, że średnia jednego bitu niezgodności ekspertów wystarcza do oszacowania częstości **znanej** awarii. Tutaj sprawdzamy, co się dzieje, gdy na wdrożeniu zawodzi inny ekspert. W ostatniej serii pojedynczy bit dał zero alarmów na 96 prób z nową silną awarią. Porównanie niezgodności między pozostałymi ekspertami wykryło ją we wszystkich 96 próbach. Dopasowanie do rozszerzonej rodziny awarii pozwoliło wybrać drzewo z testowym Brier średnio o 0,0004 powyżej najlepszego dostępnego drzewa.

## Porównanie na osobnych bankach drzew

Każdy bank zawiera po jednym zwycięskim drzewie z 12 różnych seedów rzeczywistego wyszukiwania GP. Każde wyszukiwanie trwa 10 pokoleń i używa tych samych 384 przykładów w obrębie banku. Każde powtórzenie wyboru drzewa korzysta z 256 przykładów walidacyjnych z oznaczonym kontekstem i etykietą oraz 512 nowych, nieetykietowanych predykcji docelowych. Ryzyko mierzymy na odrębnych 8192 przykładach, uśredniając dokładnie po warunkach symulatora. Predykcje źródłowe i etykiety pochodzą ze scenariusza w `fusionbench.eval.seed_selection`.

W źródle pierwszy ekspert zwraca dopełnienie swojego prawdopodobieństwa na 10% przykładów. Dwie wcześniejsze serie (6 i 8 banków) służyły do oceny dopasowania pełnego wektora średnich. W poprawnym modelu ich przybliżony test reszt zbyt często zgłaszał alarm w niektórych warunkach. Dodaliśmy test permutacyjny i zamroziliśmy go przed ostatnią serią z ośmioma nowymi seedami danych i 12 nowymi seedami GP. Łącznie uruchomiono 264 wyszukiwania. Tabele poniżej dotyczą ostatniej serii z 12 losowaniami walidacji i partii docelowej na każdy bank.

| Warunek wdrożenia | Alarm z jednego bitu | Alarm z 21 niezgodności | Alarm z 21 niezgodności i 7 miar pewności | Test niezmienności |
|---|---:|---:|---:|---:|
| Rozkład źródłowy | 0% | 8,3% | 5,2% | 3,1% |
| Częstość znanej awarii rośnie do 50% | 0% | 14,6% | 13,5% | 3,1% |
| Dodatkowy ekspert nr 3 odwraca wynik na 15% przykładów | 0% | 85,4% | 79,2% | 44,8% |
| Dodatkowy ekspert nr 2 odwraca wynik na 50% przykładów | 0% | 100% | 100% | 100% |
| Jednoczesne niezależne awarie ekspertów nr 0 i 4, obie po 50% | 0% | 99,0% | 100% | 100% |
| Osłabienie pewności eksperta nr 0 z zachowaniem jego decyzji | 0% | 8,3% | 100% | 100% |
| Identyczne predykcje, zmiana etykiety na 40% przykładów | 0% | 8,3% | 5,2% | 3,1% |

## Co sprawdzają alarmy

Test reszt dopasowuje średnie docelowe do odcinka łączącego średnie dwóch źródłowych kontekstów. Porównuje maksymalną standaryzowaną resztę z przybliżonym rozkładem Gaussa, wykorzystując niepewność wszystkich trzech prób i ponownie dopasowując częstość w każdym losowaniu. Przy jednym bicie, gdy docelowa średnia wpada między średnie źródłowe, dopasowanie ma dokładnie zerową resztę. Dodatkowe współrzędne pozwalają sprawdzać niezależne ograniczenia. Wśród nich `abs(2p−1)` dla każdego eksperta pozwala wykryć zmianę pewności przy stałych głosach.

Test niezmienności korzysta z mocniejszego założenia o jednej znanej awarii typu odwrócenie prawdopodobieństwa. Na pierwszych 128 wierszach źródłowych wybiera eksperta, którego pary wykazują największą sumę zmian niezgodności pomiędzy kontekstami. W 96 próbach wybrał właściwego eksperta za każdym razem. Niezgodności par wykluczających tego eksperta i miary pewności wszystkich ekspertów powinny mieć stały rozkład niezależnie od częstości znanej awarii. Na pozostałych 128 wierszach źródła i 512 docelowych test wykonuje 399 losowych permutacji przynależności do populacji, z progiem 0,05. Kontrola błędu testu permutacyjnego wymaga wymiennych niezależnych wierszy i poprawnie określonych niezmienników. Częstości alarmów w tabeli są opisowe, ponieważ powtórzenia korzystają z tych samych banków.

## Wybór seeda po alarmie

| Warunek | Zwykła walidacja | Dotychczasowy estymator dwóch kontekstów | Minimax dla znanej awarii | Alarm → rozszerzona rodzina | Zawsze rozszerzona rodzina |
|---|---:|---:|---:|---:|---:|
| Rozkład źródłowy | 0,0007 | 0,0008 | 0,0093 | 0,0008 | 0,0008 |
| Znana awaria, 50% | 0,0241 | 0,0019 | 0,0018 | 0,0019 | 0,0019 |
| Dodatkowy ekspert nr 2, 50% | 0,0108 | 0,0153 | 0,0313 | 0,0004 | 0,0004 |
| Jednoczesne awarie nr 0 i 4 | 0,0241 | 0,0019 | 0,0018 | 0,0019 | 0,0019 |
| Osłabienie pewności | 0,0033 | 0,0032 | 0,0036 | 0,0033 | 0,0033 |
| Ukryta zmiana etykiet | 0,0120 | 0,0124 | 0,0072 | 0,0124 | 0,0120 |

Tabela pokazuje nadwyżkę testowego Brier ponad najlepszy wynik danego banku. Minimax używa przedziału częstości znanej awarii od 10% do 50%. Kolumna z alarmem stosuje test niezmienności. Rozszerzona rodzina dopuszcza niezależną awarię jednego z sześciu dodatkowych ekspertów. Wymaga wiedzy o rodzaju awarii oraz o tym, że symulacja odwrócenia wyniku zachowuje prawdziwą etykietę. Na tych samych oznaczonych wierszach źródłowych algorytm odwraca każdy dodatkowy kanał i ponownie wylicza wyjścia wszystkich drzew. Dopasowanie momentów przechodzi po sześciu możliwych dodatkowych kanałach oraz siatce 41 częstości podstawowej awarii i analitycznie szacuje częstość dodatkowej. Etykiety wdrożeniowe nie uczestniczą w wyborze.

Przy nowej silnej awarii wybór po alarmie zmniejszył nadwyżkę Brier o 0,0149 względem dotychczasowego estymatora, z 95% przedziałem t od 0,0082 do 0,0217 dla ośmiu sparowanych średnich z banków. Względem zwykłej walidacji różnica wyniosła 0,0104 [0,0052; 0,0156]. Na tym warunku ten sam wynik dało stałe użycie rozszerzonej rodziny. Korzyść przy wyborze drzewa pochodziła z trafnej symulacji nowej awarii. Sam alarm informował o naruszeniu założeń. Dla osłabienia pewności wykrycie rozbieżności nie wystarczyło do poprawy wyboru, ponieważ rodzina dopuszczała odwrócenia głosów.

Zmiana samych etykiet zachowuje dosłownie te same wejścia obserwatora. Wszystkie jego statystyki i decyzje o alarmie pozostały identyczne w każdym powtórzeniu. Brier dotychczasowej polityki wzrósł z 0,0718 do 0,3267. Rozsądny kolejny test to porównanie małego losowego audytu z rzeczywistymi etykietami wdrożeniowymi z polityką zbierającą etykiety tylko po alarmie. Przy tej samej liczbie etykiet można zmierzyć zarówno wykrywanie ukrytej zmiany, jak i odzyskiwanie rankingu seedów.

## Odtworzenie

W katalogu zawierającym oba repozytoria na gałęzi `research/ensemble-mechanisms-stability` należy uruchomić poniższe polecenie przy zainstalowanych zależnościach FusionBench i OKAPI. Kod diagnostyki w OKAPI pochodzi z commitu `152c97d`.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.context_stress --out context-results \
    --data-seeds 800 801 802 803 804 805 806 807 \
    --gp-seeds 90 91 92 93 94 95 96 97 98 99 100 101
```

`runs.zip` zawiera konfigurację, ryzyko każdego kandydata i wyniki każdego wyboru dla wszystkich trzech serii. W dwóch pierwszych seriach rejestrowano testy reszt. W trzeciej dodano test niezmienności i odpowiadającą mu politykę po alarmie. Pełne średnie i przedziały sparowane zapisano w `scores.csv`, `alarms.csv` i `paired.csv`.
