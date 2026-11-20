# Odpowiedź na podmianę, unikalna informacja i jeden skompilowany klasyfikator

Na wszystkich 1797 przykładach cyfr ze zbioru scikit-learn udało się zastąpić uczony ensemble jednym odwzorowaniem 64 pikseli na 10 logitów i softmaxem. Największa różnica względem predykcji oryginału wyniosła 1,23 × 10⁻¹⁵. Klasyfikator tej samej wielkości nauczony bezpośrednio z etykiet osiągnął niższy błąd Briera w każdym z pięciu foldów testowych. Oba wyniki dotyczą jawnie określonej rodziny liniowej.

## Co umożliwia dokładne złożenie modeli

Wytrenowaliśmy dwóch specjalistów z regresją logistyczną. Jeden widzi lewe cztery kolumny obrazu, drugi prawe. Dla liniowego eksperta logity mają postać z = Wx+b. Logarytm prawdopodobieństwa kategorii i to zᵢ − log∑exp(zⱼ). Odjęcie średniej logarytmów po kategoriach usuwa wspólny składnik normalizujący i daje zᵢ − średnia(z). To nadal funkcja afiniczna pikseli. Następnie stosujemy standardyzację wyuczoną na treningu fuzji i uczymy liniową głowicę softmax. Złożenie map afinicznych daje jedną macierz i jeden wektor przesunięcia.

Nowa metoda `stacking_clr` przyjmuje predykcje bazowe. Jej funkcja `compile` przyjmuje macierze i przesunięcia ekspertów, wyrażone we wspólnych współrzędnych wejścia i klas. Kompilacja korzysta z wytrenowanych parametrów i nie wymaga przykładów destylacyjnych. Równość algebraiczna zakłada logity liniowe i brak numerycznego obcięcia prawdopodobieństw. Część eksperymentalna osobno mierzy wynik rzeczywistego, niezależnie wytrenowanego OKAPI z pełną standardową rodziną operatorów. Jego drzewa nie podlegają tej kompilacji.

W teście lewy ekspert był przekazany jako dwa identyczne kanały. Normalne wdrożenie oblicza jego predykcję raz. Dla dwóch ekspertów i głowicy z trzema blokami klas łączna liczba mnożeń z akumulacją w macierzach wynosi 640 + 300 = 940 na rekord. Skompilowany klasyfikator wymaga 640. Liczba parametrów afinicznych maleje z 970 do 650. Rachunek pomija funkcje aktywacji, normalizację i koszt odczytu, a więc nie jest pomiarem opóźnienia.

## Wyniki na nieużytych w uczeniu przykładach

| Metoda | Brier wieloklasowy | Trafność |
|---|---:|---:|
| Lewa połowa obrazu | 0,24284 | 84,59% |
| Prawa połowa obrazu | 0,20913 | 86,09% |
| Średnia dwóch unikalnych ekspertów | 0,17045 | 93,82% |
| OKAPI, wcześniej wskazany seed 31 | 0,08505 | 95,10% |
| Stacking na wycentrowanych logach, z kopią kanału | 0,07846 | 95,27% |
| Dokładnie złożony jeden softmax | 0,07846 | 95,27% |
| Jeden softmax z etykiet, tylko próbki ekspertów | 0,06731 | 96,16% |
| Jeden softmax z etykiet, próbki ekspertów i fuzji | 0,06122 | 96,83% |

Dwa pozostałe seedy rzeczywistego wyszukiwania GP osiągnęły Brier 0,08707 i 0,08283. We wszystkich pięciu foldach predyktor uczony bezpośrednio na próbkach ekspertów miał niższy Brier od skompilowanego o 0,00104 do 0,02172. To porównanie jednego zbioru i jednej specyfikacji. Wagi eksperta ograniczają, które kierunki wejścia widzi głowica fuzji, a każdy etap zużywa inną część danych.

## Dlaczego duża wrażliwość kanału nie ustala jego konieczności

Przy zamrożonych wagach wymieszanie predykcji jednej z dwóch kopii lewego eksperta między cyframi zwiększało Brier średnio o 0,1642, a odległość całkowitego zróżnicowania prawdopodobieństw wynosiła średnio 0,2069. Operacja tworzyła pary różnych wartości z dwóch kanałów, które normalnie zawsze są identyczne. Podstawienie dokładnego bliźniaka zachowało każdą predykcję. Jednoczesna permutacja obu kopii dawała przyrost Brier o 0,8770, a permutacja prawego kanału o 0,7504. Same liczby interwencyjne zależą od zadanej podmiany.

Usunięcie bliźniaka i ponowne uczenie dało Brier 0,07760, a maksymalna zmiana pojedynczego prawdopodobieństwa wyniosła aż 0,303. Przy karze L2 bliźniacze współczynniki mogą podzielić łączną wagę w na dwie wagi w/2. Ich łączna kara wynosi wtedy połowę kary w². Zachowując po deduplikacji jeden blok cech i mnożąc go **po standaryzacji** przez √2, odtwarzamy wcześniejszy koszt regularyzacji. Ponowne uczenie po tej korekcie odtworzyło każde prawdopodobieństwo z maksymalną różnicą 7,57 × 10⁻¹⁰. Usunięcie prawej połowy obrazu i ponowne uczenie dało Brier 0,24695. Tak można oddzielić unikalny pomiar, zależność wyuczonej funkcji i efekt regularyzacji.

## Reprodukcja i ograniczenia

Pięć stratyfikowanych foldów jest rotowanych. W każdej próbie trzy służą ekspertom, jeden fuzji i jeden ocenie. Każdy rekord dokładnie raz trafia do predykcji testowych. Zbiory treningowe nakładają się między próbami. Brak identyfikatorów piszących w tym loaderze oznacza, że podział nie sprawdza transferu na niewidzianych autorów. Plik `held_out.npz` zapisuje prawdopodobieństwa, oryginalne etykiety i fold każdego rekordu. Pole `brier` sumuje kwadraty po dziesięciu kategoriach i uśrednia po obrazach.

```bash
cd fusionbench
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.compiled_views --output docs/ensemble-study/compiled-views/results
PYTHONPATH=. python -m pytest -q tests/test_clr_compile.py
```

Kolejną granicą kompilacji są uczone przez OKAPI drzewa z operatorami minimum i maksimum, gdzie wynik może zależeć od obszaru przestrzeni surowych wejść.
