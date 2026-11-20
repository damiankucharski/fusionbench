# Kiedy 16 etykiet docelowych pomaga wybrać predyktor?

Przy słabo rozdzielonych klasach **16 losowych etykiet** zmniejszyło nadwyżkę oczekiwanego Brier z **0,04596 do 0,01554**. Różnica wyniosła **0,03043** z 95-procentowym przedziałem od 0,01486 do 0,04600 w 24 nowych bankach. Przed etykietowaniem i po nim uśrednialiśmy te same reguły na tych samych wejściach. Zmieniły się wyłącznie wagi nadane kandydatom na podstawie trafności ich prognoz klas.

Przy silnym sygnale model nauczony z samych wejść osiągał Brier ponad optimum równy 0,00050. Etykiety praktycznie nie zmieniły tego wyniku. W dwóch światach o identycznych pomiarach i różnych proporcjach klas prosty licznik 16 etykiet wypadł lepiej od kombinacji modeli.

## Jak porównujemy kandydatów

Dla każdego banku dopasowaliśmy modele do 1024 anonimowych obserwacji i zmierzyliśmy ich logarytmiczną gęstość na kolejnych 1024. Cztery starty EM i sześć zapisanych momentów treningu dają 24 klasyfikatory logistyczne na sześciu surowych wejściach. Dodaliśmy 19 stałych predykcji w modelu dopuszczającym brak sygnału klasowego. Wszystkie stałe dzielą jeden rozkład wejść i łącznie otrzymują połowę początkowej masy. Druga połowa przypada po równo czterem trajektoriom EM oraz kolejnym punktom w obrębie każdej trajektorii.

Po odsłonięciu etykiet mnożymy wagę kandydata przez prawdopodobieństwa, jakie przypisał zaobserwowanym klasom na losowych przykładach. Uśrednienie korzysta także z wcześniejszej oceny gęstości wejść. Osobno sprawdzamy uśrednienie wykorzystujące tylko etykiety, wybór jednego modelu według Brier, korektę wyrazu wolnego zwycięzcy gęstości i estymację stałego prawdopodobieństwa (liczba dodatnich etykiet + 1)/(liczba etykiet + 2).

| Słaby sygnał, nadwyżka oczekiwanego Brier | 0 etykiet | 4 | 16 | 64 |
|---|---:|---:|---:|---:|
| Uśrednianie z gęstości i późniejszych etykiet | 0,04596 | 0,02684 | **0,01554** | **0,00905** |
| Uśrednianie oparte tylko na etykietach i początkowych masach | 0,04648 | 0,02736 | 0,02018 | 0,01002 |
| Jeden kandydat według gęstości, z korektą wyrazu wolnego | 0,03731 | 0,03754 | 0,02826 | 0,02173 |
| Jeden kandydat według Brier na etykietach | 0,03731 | 0,06623 | 0,02963 | 0,01147 |
| Stała ze zliczenia etykiet | 0,13324 | 0,09671 | 0,08054 | 0,07362 |

Przy czterech etykietach twardy wybór kandydata zwiększył błąd z 0,03731 do 0,06623. Przy 16 etykietach uśrednianie z gęstości i etykiet uzyskało nadwyżkę mniejszą o 0,01273 od korekty wyrazu wolnego, z przedziałem od 0,00389 do 0,02157. Przewaga nad uśrednianiem wykorzystującym same etykiety wyniosła 0,00465, z przedziałem od −0,00044 do 0,00974. Rzeczywisty posterior pozwoliłby retrospektywnie wskazać jednego kandydata ze stratą 0,00682. Ten punkt odniesienia wymaga wiedzy testowej.

Reguła uśredniająca przypisywała modelom stałym średnio 40,0% masy przed etykietami i 16,8% po 16 etykietach. Średnia prognoza udziału klasy wzrosła z 0,640 do 0,738 przy rzeczywistej wartości 0,750. Różnicę względem korekty wyrazu wolnego mogą wyjaśniać zarówno zmiany kierunków wag, jak i mniejsza wariancja prognoz dzięki uśrednianiu.

## Te same wejścia mogą wymagać innych odpowiedzi

Przy braku sygnału użyliśmy dokładnie tych samych macierzy wejść, tej samej wytrenowanej puli i tych samych liczb losowych w dwóch światach. W jednym klasa dodatnia ma udział 25%, w drugim 75%. Zbieżność prognoz przed otrzymaniem etykiet sprawdziliśmy dla każdego z 24 odłożonych banków. W obu światach uśrednianie wyznaczyło na starcie średnie prawdopodobieństwo 0,498. Po 16 etykietach wyniosło odpowiednio 0,281 i 0,704.

| Warunek, 16 etykiet | Uśrednianie bez etykiet | Uśrednianie z etykietami | Jedna sigmoida po kalibracji wyrazu wolnego | Stała ze zliczenia etykiet |
|---|---:|---:|---:|---:|
| Słaby sygnał | 0,04596 | **0,01554** | 0,02826 | 0,08054 |
| Silny sygnał, jedna zależna para czujników | 0,00050 | **0,00050** | 0,00323 | 0,17305 |
| Brak sygnału, 25% klasy dodatniej | 0,06417 | 0,01240 | 0,03277 | **0,01064** |
| Brak sygnału, 75% klasy dodatniej | 0,06655 | 0,01341 | 0,03544 | **0,01085** |

Gdy prawdziwe prawdopodobieństwo jest stałe, licznik etykiet wykorzystuje poprawną strukturę zadania. Jego dokładna oczekiwana nadwyżka Brier przy n etykietach i udziale p wynosi [np(1−p)+(1−2p)²]/(n+2)². Dla n=16 oraz p=0,25 lub p=0,75 daje to 0,01003, blisko uzyskanych wartości empirycznych. Gdy czujniki faktycznie rozróżniają klasy, ten sam licznik traci informację zawartą w ich obserwacjach.

## Co można wdrożyć

Wybór pojedynczego modelu i korekta wyrazu wolnego mieszczą się w jednej sigmoidzie o sześciu wagach i wyrazie wolnym lub w stałej. Arytmetyczna średnia prawdopodobieństw wielu sigmoid wymaga dodatkowych obliczeń. Dla słabego sygnału i budżetu 16 etykiet efektywna liczba kandydatów wyliczona jako 1/Σw² wynosiła 6,07. Ta liczba opisuje koncentrację wag. W tym rozwiązaniu dokładna predykcja może nadal wymagać wykonania wszystkich 24 sigmoid, a wkłady 19 stałych można zsumować raz. Osobnym pytaniem jest to, czy uśrednienie parametrów liniowych zachowa zysk po skompilowaniu do jednej sigmoidy.

Badanie wykorzystuje przygotowane mieszaniny Gaussa o znanej orientacji klas. Każdy wynik banku jest średnią z 16 odrębnych losowań etykiet. Budżety w jednym losowaniu są zagnieżdżone. Miary oceniamy na 8192 kolejnych wejściach, korzystając z prawdziwego posterioru wyłącznie podczas oceny. Cztery banki rozwojowe i 24 potwierdzające są rozłączne. Przedział t dla głównego efektu wykorzystuje 24 niezależne banki. Obliczenia trwały 13,77 s na lokalnym CPU.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m fusionbench.eval.mixture_sentinels --out mixture-sentinels-results
```

Pliki banks.csv i sentinel_draws.csv.gz zawierają wyniki wszystkich banków i powtórzeń. [Protokół sprzed obliczeń](https://github.com/damiankucharski/fusionbench/blob/d57a3b7/docs/ensemble-study/mixture-sentinels/protocol.md)
