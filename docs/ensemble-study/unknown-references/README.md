# Której referencji można zaufać?

Dwie sprzeczne referencje mogą dawać dokładnie ten sam rozkład wszystkich obserwacji w dwóch światach, w których poprawną decyzję podejmują przeciwni eksperci. Trzecia referencja pomogła, gdy najwyżej jeden z trzech kanałów powielał błędy słabszego eksperta. Przy dwóch niezależnych referencjach o sile 0,40 i jednym kopierze mediana ocen kontekstu dała Brier 0,1375. Połączenie wszystkich obserwacji przed oszacowaniem kontekstu dało 0,2242.

## Granica identyfikowalności

Niech A, B będą głównymi czujnikami o siłach 0,85 i 0,25 względem zbalansowanej etykiety. W pierwszym świecie H jest wiarygodną referencją o sile 0,75, a D kopiuje słabszy B według mechanizmu ustalonego w poprzednim eksperymencie. W drugim świecie zamieniamy jednocześnie role A z B oraz H z D. Parametry kopiowania zrównują wszystkie sześć obserwowalnych momentów par w obu światach. Momenty nieparzyste znikają z symetrii. Moment czwartego rzędu E[ABHD] zachowuje wartość przy tej zamianie. Zatem każda z 16 czwórek znaków ma to samo prawdopodobieństwo w obu światach. Kod wylicza całą tablicę bez losowania.

Również przy losowym, stałym w partii przypisaniu jedynej wadliwej referencji do jednego z dwóch identyfikatorów można sparować każdą konfigurację z nierozróżnialną konfiguracją w przeciwnym świecie. Przy równej częstości światów selektor oparty wyłącznie na nieetykietowanych danych nie może podnieść oczekiwanej trafności ponad 77,5% w tym konstrukcie. Gdy wiadomo, który identyfikator odpowiada sprawnej referencji, ta dodatkowa informacja usuwa dwuznaczność.

## Selekcja rzeczywistych drzew OKAPI

Każda z 12 nowych prób obejmowała dwa wyszukiwania GP oraz odrębne dane do estymacji ryzyka drzew i średnich niezgodności. Wszystkie 24 wyszukiwania wybrały liść odpowiadający ekspertowi silniejszemu w źródłowym reżimie. Dwa źródłowe reżimy kalibracyjne dostarczały po 384 oznaczone przykłady. W docelowym teście losowaliśmy po 200 partii na reżim. Tożsamość zawodnego obserwatora była losowa między partiami i stała wewnątrz danej partii.

Porównaliśmy rzutowanie uśrednionego wektora niezgodności na odcinek źródłowych średnich ze średnią lub medianą osobnych oszacowań częstości. Każde osobne oszacowanie jest ograniczone do [0,1]. Kandydat z najniższym kalibrowanym ryzykiem przy tej częstości wygrywa selekcję. Dane do porównania metod oraz kolejność sekwencji losowych są wspólne.

| Trzy referencje, 32 przykłady | Brier po połączeniu obserwacji | Średnia osobnych oszacowań | Mediana osobnych oszacowań |
|---|---:|---:|---:|
| Wszystkie sprawne o sile 0,75 | 0,0925 | 0,0925 | 0,0927 |
| Jedna pozoruje przeciwny reżim, dwie sprawne o sile 0,75 | 0,1139 | 0,1061 | 0,0974 |
| Wszystkie sprawne o sile 0,40 | 0,1024 | 0,1028 | 0,1052 |
| Jedna zawsze kopiuje słabszego eksperta, dwie sprawne o sile 0,40 | 0,2242 | 0,1490 | 0,1375 |
| Dwie są duplikatami tego samego kopiera | 0,2540 | 0,2545 | 0,2550 |

W zaplanowanym porównaniu z jedną referencją kopiującą słabego eksperta różnica Brier mediana minus łączenie obserwacji wyniosła −0,0867 z przedziałem 95% [−0,0967, −0,0766] po 12 sparowanych próbach. Stała równa mieszanina prawdopodobieństw obu drzew dała 0,1446. W tym scenariuszu dwa sprawne kanały mają oczekiwaną różnicę zgodności po 0,40 × (0,85 − 0,25) = 0,24, a kopier wnosi przeciwną różnicę 1 − 0,85 × 0,25 = 0,7875. Zwiększenie próby poprawia precyzję obciążonej średniej w niewłaściwą stronę. Przy 128 przykładach jej trafność spadła do 65,30%, a trafność mediany wzrosła do 90,74%.

Mediana wykorzystuje założenie o większości sprawnych źródeł. Kopia tego samego błędu z dwoma identyfikatorami łamie to założenie. Na zdrowych danych o małej sile medianę obciąża koszt większej zmienności statystycznej. Zjawisko badano w jednej rodzinie rozkładów z dwoma podstawowymi ekspertami. Zamrożony protokół i ryzyka dla każdej próby udostępniamy obok raportu.

## Odtworzenie

Nowa funkcja OKAPI `estimate_context_prevalences` oblicza oszacowania dla wielu partii i obserwatorów, zachowując osobną wartość dla każdego. Udostępnia ją commit `abcdcca` na gałęzi badawczej OKAPI. Polecenia zakładają checkouty obu projektów obok siebie.

```bash
cd fusionbench
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.unknown_references --output docs/ensemble-study/unknown-references/results
PYTHONPATH=.:../okapi python -m pytest -q tests/test_unknown_references.py ../okapi/test/test_context_batches.py ../okapi/test/test_selection.py
```

Przy dwóch referencjach mediana dwóch oszacowań jest ich średnią arytmetyczną. Wyniki w `risks.csv` rozdzielają oba reżimy i seedy. `four_bit_mass.csv` zawiera ich dokładne rozkłady obserwacji dla wskazanych w dowodzie tożsamości referencji. Następne pytanie dotyczy kontrolowanej podmiany wejścia u obserwatora, który potencjalnie korzysta z predykcji innego modelu.
