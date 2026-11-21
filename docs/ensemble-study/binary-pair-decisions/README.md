# Dwie binarne decyzje w jednym modelu bez destylacji

Arytmetyczna średnia prawdopodobieństw dwóch równie ważonych binarnych ekspertów przekracza 0,5 dokładnie wtedy, gdy suma ich logitów jest dodatnia. Jeśli oba logity są afiniczne w tych samych surowych cechach, można więc uśrednić współczynniki i bias i wdrożyć jeden klasyfikator. **Zgodność klasy jest globalna i obejmuje każdy możliwy wektor wejściowy w arytmetyce rzeczywistej.** W 54 360 porównaniach para–rekord z UCI Spambase i UCI HAR wszystkie klasy były zgodne również w obliczeniach float64. Dwa zbiory zawierają łącznie 1812 odłożonych rekordów, wykorzystywanych wielokrotnie przez pary i układy widoków.

Zależność wynika z tożsamości σ(a)+σ(b)−1=σ(a)−σ(−b) i ścisłej monotoniczności sigmoidy. Przy arytmetycznej średniej prawdopodobieństw równej progowi zachowujemy standardowy dla tego badania wybór klasy 1 przy wyniku 0,5. Same prawdopodobieństwa mogą się zmienić. W rozłącznym Spambase RMSE do wybranej pary wyniosło 0,09067, mimo identycznych decyzji. [Dowód, kryterium dla większych zespołów i kontrprzykłady](proof.md)

## Czy wybrana para była użyteczna?

Każdy ekspert miał dostęp do przydzielonego zestawu cech. Spośród 15 par wybieraliśmy najmniejszy walidacyjny Brier arytmetycznej średniej. Dostęp do etykiet eksperta, pary i pełnej regresji był taki sam. Poniższe wyniki to trafność na odłożonym zbiorze, w HAR ważona jednakowo po dziewięciu osobach.

| Dane i widoki | Wybrana para oraz jej pojedynczy odpowiednik | Najlepszy składnik z walidacji | Pełna surowa regresja | Silna pojedyncza regresja |
|---|---:|---:|---:|---:|
| Spambase, rozłączne | 89,58% | 83,93% | 91,86% | 93,92% |
| Spambase, zachodzące | 92,07% | 91,86% | 91,86% | 93,92% |
| HAR schody, rozłączne | 96,99% | 94,19% | 98,02% | 98,02% |
| HAR schody, zachodzące | 98,02% | 98,03% | 98,02% | 98,02% |

Na rozłącznym Spambase para zyskała 5,65 punktu procentowego [3,52; 8,10] wobec wybranego składnika, ale straciła 4,34 punktu [2,49; 6,39] wobec silnej pełnej regresji z walidacyjnym wyborem transformacji log1p. Z kolei średni Brier pary 0,09487 po kompresji wyniósł 0,08533, podczas gdy silna pojedyncza regresja uzyskała 0,04358. Dla rozłącznego HAR różnica trafności pary względem składnika wyniosła +2,80 punktu [−0,10; +6,81]. Przedziały dotyczą już dopasowanych modeli i zmienności odłożonych grup, przy 4000 powtórzeń bootstrapu. Te same podziały były używane we wcześniejszym badaniu kompresji i nie stanowią nowej replikacji na kolejnych osobach.

## Czy trzy sumy wystarczają zamiast sześciu głosów?

Dla sześciu równie ważonych ekspertów dzielimy je na trzy pary i sprawdzamy znaki trzech średnich logitów. Jeżeli wszystkie pary wskazują tę samą stronę progu (z obsługą zer), decyzja całej puli jest rozstrzygnięta. W pozostałych przypadkach reguła oblicza po jednej półróżnicy logitów na parę i odzyskuje sześć głosów. Wariant ustalony na źródle wybiera spośród 15 możliwych parowań największe pokrycie na wejściach walidacyjnych bez odczytu ich etykiet. Wszystkie poniższe procedury zwróciły identyczne ostateczne klasy co sześć pełnych głosów.

| Dane i widoki | Jednomyślność ekspertów | Pary w kolejności indeksów | Parowanie wybrane na źródle | Sortowanie sześciu logitów na każdym wejściu |
|---|---:|---:|---:|---:|
| Spambase, rozłączne | 53,53% | 75,79% | **77,85%** | 89,79% |
| Spambase, zachodzące | 89,58% | 94,68% | 95,98% | 98,26% |
| HAR, rozłączne | 88,18% | 94,48% | 93,55% | 97,81% |
| HAR, zachodzące | 98,69% | 99,79% | 99,67% | 100,00% |

W głównym warunku Spambase przyrost pokrycia wyniósł **24,32 punktu procentowego [19,92; 28,59]**. Przy zmianie osoby w HAR zwykła kolejność indeksów uzyskała nieco większe pokrycie niż parowanie wybrane na źródle. Sortowanie na każdym wejściu znajduje najlepsze możliwe potwierdzenie tego rodzaju i wymaga wszystkich logitów. Dla stałego parowania da się wyliczać trzy średnie jednym przekształceniem surowych cech, a półróżnice tylko dla nierozstrzygniętych wejść.

Koszt zależy od rozkładu cech. Przy rozłącznych widokach Spambase średnia liczba sigmoid spadła z 6 do 1,33, lecz przy wykorzystaniu zer liczba MAC wzrosła z 57 do 69,63 na przykład. Przy zachodzących widokach Spambase spadła z 285 do 177,87 MAC, a liczba sigmoid z 6 do 0,24. W zachodzącym HAR liczba MAC spadła z 2805 do 1688,51, przy średnio 0,02 sigmoid na przykład. To liczba niezerowych iloczynów–akumulacji w regułach afinicznych. Doliczenie obsługi warunków, odczytu parametrów i sortowania zależy od wykonania. Nie przeprowadzaliśmy pomiaru przyspieszenia na sprzęcie.

## Zakres dokładności i odtworzenie

Samo uśrednienie prawdopodobieństw nie zawsze prowadzi do jednej hiperpłaszczyzny. Para z wagami 0,55 i 0,45 może przeciąć próg trzy razy. To samo może nastąpić dla trzech ekspertów o równych wagach. W kontroli nierównych wag przecięcia wystąpiły przy −2,30259, 0 i +2,30259. Z drugiej strony nawet duża pula zachowuje jedną globalną granicę, jeśli logity środkowe wszystkich par są dodatnimi wielokrotnościami tej samej funkcji afinicznej. Każde z 924 możliwych posortowanych zestawów sześciu logitów na siedmiopunktowej siatce dało taki sam wynik testu istnienia zgodnego parowania dla sortowania i przeszukania wszystkich 15 możliwości.

Pierwsze pełne obliczenie trwało 6,54 s na CPU, a powtórzenie po uporządkowaniu skryptu 5,55 s. Wszystkie pliki wyników oprócz metadanych czasu były bajtowo identyczne. Od pierwszego sprawdzenia celu do gotowego raportu upłynęło 370 s.

Gotowe wektory wag dwóch i sześciu komponentów oraz ich grupowania znajdują się w results/models.json.gz. Niezależne wczytanie pliku odtworzyło wszystkie cztery pary wyników trafności z tolerancją 10⁻¹². Wiersze results/pairs.csv dokumentują każdą z 15 par, a results/matchings.csv każde z 15 pełnych parowań. W natywnej funkcji OKAPI trudne przypadki blisko maszynowego remisu są sprawdzane z uwzględnieniem symetrii i znaku logarytmicznej sumy, co obejmuje nasycone sigmoidy.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python fusionbench/examples/binary_pair_decisions.py --spambase spambase.zip --har har.zip --out pair-results
```

[Przedrejestracja](https://github.com/damiankucharski/fusionbench/blob/e36c3b1/docs/ensemble-study/binary-pair-decisions/protocol.md) · [Implementacja OKAPI](https://github.com/damiankucharski/OKAPI/blob/a7c23e8/okapi/binary_pairs.py) · [Poprzedni test użyteczności na tych samych grupach](https://github.com/damiankucharski/fusionbench/blob/dd30903/docs/ensemble-study/real-sigmoid-pool/README.md)
