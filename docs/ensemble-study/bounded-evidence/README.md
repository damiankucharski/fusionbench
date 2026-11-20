# Kiedy wystarczą dwa neurony ReLU?

W scenariuszu `polarity_fault` jeden czujnik niezależnie odwraca znak pomiaru na 10% przykładów. Szum sześciu pozostałych czujników ma korelację 0,65. Pojedynczy model z jednym ograniczeniem amplitudy surowego wejścia osiągnął w niezależnym potwierdzeniu **93,36% trafności**. Reguła Bayesa znająca mechanizm osiągnęła 93,59%, model liniowy 92,90%, zwykłe MLP 92,89%, a najlepsze walidacyjne drzewo z ośmiu wyszukiwań OKAPI 91,12%. Pojedyncza sieć uczyła się na surowych pomiarach i prawdziwych etykietach. Nie wymagała destylacji, predykcji ekspertów ani identyfikatora wadliwego czujnika podczas nauki lub wdrożenia.

## Dlaczego ograniczenie ma sens

Dla pierwszego pomiaru bez awarii ekspert zwraca prawdopodobieństwo p. Gdy awaria odwraca pomiar z prawdopodobieństwem π, a klasy mają równe udziały i symetryczne rozkłady, prawdziwe prawdopodobieństwo z tego pojedynczego pomiaru wynosi π + (1−2π)p. Jego wkład w logarytm ilorazu szans ma zatem bezwzględną wartość najwyżej log((1−π)/π). Przy 10% awarii granica wynosi log(9) ≈ 2,20: nawet ekstremalny pomiar nie dostarcza nieograniczonego dowodu.

Pozostałe pomiary x, warunkowo na klasę, są gaussowskie ze średnimi ±s i wspólną macierzą kowariancji Σ. Ich łączny wkład wynosi 2 sᵀΣ⁻¹x. Przy korelacji 0,65 dwa najsłabsze stabilne czujniki dostają współczynniki −0,50 i −1,33, choć każdy jest trafniejszy od losowego na własną rękę. Przy ustalonych pozostałych pomiarach pomagają oszacować wspólny składnik szumu. Sumę logarytmicznych wkładów obu bloków sprawdziliśmy numerycznie przez bezpośrednie porównanie gęstości mieszanin Gaussa.

Uczący się model przycina jedno wybrane wejście w symetrycznym przedziale i stosuje regresję logistyczną do wszystkich siedmiu pomiarów. Funkcję przycięcia można zapisać jako ReLU(x+t) − ReLU(x−t) − t. Model ma siedem współczynników liniowych i wyraz wolny, a walidacja wybiera indeks czujnika i jeden z siedmiu progów lub brak przycięcia. W głównym warunku wybrała właściwy czujnik w 8 z 8 nowych banków. Wyuczona maksymalna amplituda jego wkładu w logarytm szans wahała się od 1,62 do 2,29. Przy niezależnym szumie stabilnych czujników ten sam wybór trafił we właściwy kanał tylko w 3 z 8 banków.

## Kontrole i koszt

| Model | Nadwyżka oczekiwanego Brier przy obu zjawiskach | Trafność |
|---|---:|---:|
| Dokładna reguła Bayesa | 0 | 93,59% |
| Wybrane ograniczenie amplitudy, dwie ReLU | 0,00160 | 93,36% |
| Liniowy na siedmiu pomiarach | 0,00485 | 92,90% |
| MLP wybrane z ośmiu konfiguracji/startów | 0,00536 | 92,89% |
| Przycięcie dozwolone tylko na błędnie wskazanym kanale | 0,00581 | 92,75% |
| Najlepsze z ośmiu krótkich wyszukiwań OKAPI | 0,02336 | 91,12% |
| Reguła Bayesa ignorująca awarię | 0,03076 | 90,79% |
| Reguła Bayesa ignorująca korelację | 0,05980 | 87,21% |

Pierwotna różnica wobec regresji liniowej wyniosła **0,00325 [0,00293; 0,00357]** nadwyżki Brier, z 95% przedziałem t z ośmiu sparowanych banków. Znając prawdziwy posterior q, nadwyżkę oczekiwanego Brier modelu p mierzymy na 16 384 nowych wejściach jako średnią (p−q)². Po wyłączeniu awarii i pozostawieniu korelacji wynik liniowy był lepszy: 0,00074 wobec 0,00114 po poszukiwaniu przycięcia. Przy stałym budżecie walidacji możliwość wyboru dodatkowego parametru ma koszt.

W każdej konfiguracji użyto 768 etykiet do dopasowania i 256 do wyboru wariantu, a całe badanie zawierało 384 wyszukiwania GP po 10 pokoleń. OKAPI korzystało z symulowanych prawdopodobieństw ekspertów. Trening wariantu z przycięciem przeszukał 50 tanich modeli logistycznych, więc równa liczba etykiet nie oznacza równego nakładu optymalizacji. Dane mają jednoznacznie określony mechanizm i tylko siedem surowych cech. Wartość tej reprezentacji na rzeczywistych danych pozostaje otwarta.

Praktyczne pytanie na następną rundę dotyczy przenoszenia ujemnych wag stabilnych ekspertów. Po zaniku wspólnego szumu ich znak powinien się zmienić. Sprawdzimy, czy można oszacować tę zmianę z nieetykietowanych predykcji docelowych i jak odróżnić zmianę korelacji od zmiany proporcji klas.

## Użycie

Scenariusz jest dostępny w standardowym rejestrze FusionBench jako `polarity_fault`. Funkcja `PolarityFault.bayes_oracle` oblicza posterior z samych predykcji i parametrów generujących bez użycia prawdziwych etykiet poszczególnych rekordów.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.bounded_evidence --out bounded-results
```

Gotowe parametry 48 dopasowanych małych sieci zapisano w `cap_models.json`. Funkcja `fusionbench.eval.bounded_evidence.predict_raw_cap(x, **record["parameters"])` przyjmuje macierz siedmiu surowych pomiarów. Weryfikacja eksportu odtworzyła Brier każdego z 48 modeli z tolerancją 10⁻¹². Obliczenia badania trwały 56,27 s, dodatkowa kontrola eksportu 4,61 s. [Protokół sprzed obliczeń](https://github.com/damiankucharski/fusionbench/blob/1306ace/docs/ensemble-study/bounded-evidence/protocol.md)
