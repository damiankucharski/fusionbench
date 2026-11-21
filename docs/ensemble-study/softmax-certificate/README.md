# Jedna sieć wieloklasowa z gwarancją odległości od puli

Jeżeli pięciu ekspertów zwraca softmaksy liniowych logitów na tych samych surowych wejściach, można z ich parametrów zbudować jeden softmax i obliczyć górną granicę **całkowitej wariacji** względem arytmetycznej średniej ich prawdopodobieństw. Granica obowiązuje punktowo, obejmuje różnicę prawdopodobieństwa dowolnego zbioru klas, a jej ostra stała nie rośnie z liczbą klas. Na rzeczywistych danych ograniczenie dobrze rozpoznawało niemal identycznych ekspertów. Dla odmiennych ekspertów bywało zachowawcze nawet wtedy, gdy wszystkie obserwowane decyzje puli i pojedynczego modelu były zgodne.

Niech zᵢ będzie wektorem logitów i-tego modelu na danym przykładzie, aᵢ jego udziałem, a z̄ średnią ważoną logitów. Oznaczmy przez osc rozstęp współrzędnych wektora. Dla P=Σaᵢ softmax(zᵢ) oraz Q=softmax(z̄),

```text
TV(P,Q) ≤ min(1, Σᵢ aᵢ osc(zᵢ−z̄)² / (12√3)).
```

OKAPI oferuje również oszacowanie oparte na jednej kwadratowej funkcji surowego wejścia. Może ją wykonać po usunięciu źródłowych ekspertów. Wysoki wymiar zwiększa koszt przechowania i obliczenia tej macierzy. Zwykła predykcja skompresowanego modelu potrzebuje tylko jednej macierzy wag i biasu. Kompresja nie korzysta z predykcji nauczyciela do treningu. [Dowód ostrej stałej, własności granicy Brier i kontrprzykład dokładnej redukcji](proof.md)

## Jak często certyfikat rozstrzyga pytanie na prawdziwych danych?

Test objął 360 odłożonych przykładów klasyfikacji 10 cyfr i dziewięć odłożonych osób wykonujących sześć aktywności w UCI HAR. Pięć regresji wieloklasowych miało cechy rozłączne, mocno zachodzące, niemal identyczne lub identyczne. Dokładnie identyczne modele dały zgodne predykcje i zerowe granice na obu zbiorach.

| Dane i widoki | Średni rzeczywisty TV | Faktycznie TV ≤0,05 | Gwarancja TV ≤0,05 z oscylacji | Gwarancja wspólnej top-1 z oscylacji | Faktyczna zgodność top-1 |
|---|---:|---:|---:|---:|---:|
| Cyfry, rozłączne | 0,19857 | 3,06% | 0% | 0% | 96,39% |
| Cyfry, zachodzące | 0,01199 | 93,61% | 0% | 37,22% | 100% |
| Cyfry, niemal identyczne | 0,00026 | 100% | 99,17% | 99,72% | 100% |
| HAR, rozłączne | 0,03918 | 71,32% | 0% | 0,61% | 99,40% |
| HAR, zachodzące | 0,00797 | 99,37% | 0,16% | 77,36% | 100% |
| HAR, niemal identyczne | 0,00001 | 100% | 100% | 100% | 100% |

W zachodzących widokach HAR surowsza granica kwadratowa certyfikowała wspólną decyzję na 62,12% pomiarów, a zgodność TV≤0,05 na 0%. Gwarancja decyzji wymaga, by przewaga najlepszego prawdopodobieństwa Q nad drugim przekraczała dwukrotność granicy TV. Żaden certyfikowany przykład nie zmienił zwycięskiej klasy. Dodanie jednemu ekspertowi tej samej funkcji afinicznej do logitu każdej klasy zmieniło wartości obu certyfikatów najwyżej o 2,5×10⁻¹⁴ w arytmetyce zmiennoprzecinkowej.

## Czy średnia ekspertów miała coś do zaoferowania?

| Dane i widoki | Sumaryczny Brier pełnej puli | Jeden skompresowany softmax | Najlepszy składnik wybrany na walidacji | Jedna logistyka wszystkich cech |
|---|---:|---:|---:|---:|
| Cyfry, rozłączne | 0,21343 | 0,10689 | 0,27787 | 0,05717 |
| Cyfry, zachodzące | 0,05916 | 0,05754 | 0,07098 | 0,05717 |
| Cyfry, niemal identyczne | 0,05616 | 0,05613 | 0,05447 | 0,05717 |
| HAR, rozłączne | 0,10119 | 0,09316 | 0,13621 | 0,08402 |
| HAR, zachodzące | 0,08109 | 0,08051 | 0,07708 | 0,08402 |
| HAR, niemal identyczne | 0,08415 | 0,08416 | 0,08427 | 0,08402 |

Na rozłącznych widokach oba zespoły poprawiły wynik najlepszego składnika, lecz bezpośrednia regresja wszystkich cech uzyskała niższy Brier. Przy zachodzących widokach HAR pełna pula poprawiła Brier względem pełnej logistyki o 0,00294 [0,00129; 0,00490] w grupowym bootstrapie dziewięciu osób. Jeszcze lepszy był tam wybrany walidacyjnie pojedynczy członek. Te dane rozdzielają trzy pytania: czy pojedynczy model utrzymuje decyzje zespołu, czy odtwarza prawdopodobieństwa oraz czy zespół wnosi przewagę nad konkurencyjnym modelem z tym samym budżetem etykiet. Przedziały dotyczą gotowych dopasowań i zmienności grup testowych.

## Użycie

```python
from okapi.softmax_pool import compile_softmax_pool

model = compile_softmax_pool(W, biases, ensemble_weights, keep_sources=False)
probability = model.predict(X)
tv_upper_bound = model.error_bound(X)
```

FusionBench udostępnia tę samą funkcję na standardowych metodach `SimpleAverage` oraz `ValWeighted`. Druga wymaga wcześniejszego dopasowania na predykcjach walidacyjnych. Opcja instalacyjna `pip install ".[okapi]"` w katalogu repozytorium wskazuje konkretną rewizję OKAPI zawierającą kompilator. Dla dopasowanych regresji logistycznych scikit-learn albo potoków StandardScaler + LogisticRegression wystarczy:

```python
from fusionbench.methods.averaging import SimpleAverage

fused = SimpleAverage().compile_estimators(estimators)
p = fused.predict(raw_X)
labels = fused.predict_labels(raw_X)
tv_bound = fused.error_bound(raw_X)
```

Dla różnych podzbiorów cech ustaw `feature_indices` i `n_features` we wspólnej surowej przestrzeni. Można też podać od razu macierze przez `compile(W, biases)`. Wykonywanie `predict` respektuje tę samą normalizację co pozostałe metody FusionBench. Obcinanie prawdopodobieństw od dołu, a następnie renormalizacja nie powiększają TV. Istotnie, niech A i B będą sumami obu wektorów po obcięciu. Obie są ≥1. Jeśli A≥B, dodatnie różnice po renormalizacji są ograniczone przez dodatnie różnice po obcięciu podzielone przez A. Obcięcie jest niemalejące i nierozszerzające w każdej współrzędnej. Łącznie suma dodatnich różnic nie rośnie. Dla A<B wystarczy zamienić argumenty.

Wersja z keep_sources=True umożliwia dodatkowo model.oscillation_bound(X). Macierz W ma wymiary eksperci × klasy × cechy. Warunkowanie na wejścia umożliwia wykonywanie osobnych decyzji co do dopuszczalnej niezgodności. Na ten moment nie zmierzono przewagi czasowej wersji z certyfikatem. Kolejny test sprawdzi, czy niskorzędowa postać tej samej macierzy i sprawdzanie marginesów logitów podnoszą pokrycie gwarancji decyzji przy rozsądnym koszcie obliczeń.

Realny benchmark trwał 238,99 s na jednym rdzeniu CPU. Pliki results/results.csv, results/contrasts.csv i results/test_examples.csv.gz zawierają podsumowania i poszczególne wyniki. Pełne testy gałęzi OKAPI zakończyły się wynikiem 326 zaliczonych i 1 pominiętym w 15,37 s.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python fusionbench/examples/real_softmax_certificate.py --har har.zip --out softmax-results
```

[Przedrejestracja matematyczna](https://github.com/damiankucharski/fusionbench/blob/1e0eb8e/docs/ensemble-study/softmax-certificate/protocol.md) · [Przedrejestracja benchmarku](https://github.com/damiankucharski/fusionbench/blob/1e0eb8e/docs/ensemble-study/softmax-certificate/benchmark-protocol.md) · [Metoda OKAPI](https://github.com/damiankucharski/OKAPI/blob/f8c77ca/okapi/softmax_pool.py)
