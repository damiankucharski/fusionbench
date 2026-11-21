# Jedna błędna etykieta zmienia znaczenie gwarancji

**Zbiór wszystkich monotonicznych reguł o minimalnej liczbie błędów nadal ma dwie dopuszczalne skrajności**, nawet jeśli etykiety są zaszumione. Każda obserwacja wnosi koszt f(x) lub 1−f(x), więc suma strat dwóch funkcji jest równa sumie strat ich minimum i maksimum. Jeżeli obie były optymalne, ich minimum i maksimum także muszą być optymalne. W OKAPI jedno przecięcie w grafie porządku wyznacza liczbę błędów i obie skrajne prognozy. Maksymalny wymiar tej pełnokostkowej implementacji to 12 wejść. [Wyprowadzenie](derivation.md) · [Algorytm minimalnego przekroju](https://github.com/damiankucharski/OKAPI/blob/f850e5f/okapi/monotone_erm.py)

Gwarancja wobec wszystkich minimalizatorów empirycznych nie obejmuje automatycznie prawdy. W 12 niezależnych panelach losowaliśmy po sześć prawdziwych reguł czterech bitów i zmienialiśmy dokładnie jedną końcową etykietę przy każdym z dwóch budżetów. W każdym ze 144 dopasowań oba końce i liczba błędów były identyczne z pełnym przeglądem wszystkich 166 funkcji. W wariancie z 12 etykietami prawda znalazła się jednak pośród minimalizatorów tylko w 51 z 72 zadań.

| Etykiety | Zbiór możliwych modeli | Prawda w zbiorze | Obie skrajności w zbiorze | Najgorszy Brier środka skrajności | Optymalny najgorszy Brier | Brier tego optimum wobec prawdy |
|---:|---|---:|---:|---:|---:|---:|
| 12 | Najmniej błędów treningowych | 70,83% | 100% | 0,08160 | 0,08160 | 0,11372 |
| 12 | Najwyżej jeden błąd | 100% | 80,56% | 0,10959 | 0,10575 | 0,10228 |
| 32 | Najmniej błędów treningowych | 90,28% | 100% | 0,04948 | 0,04948 | 0,05729 |
| 32 | Najwyżej jeden błąd | 100% | 93,06% | 0,05490 | 0,05418 | 0,05392 |

**Reguły dopuszczające jeden błąd mogą nie zawierać swojego minimum lub maksimum.** Przy 12 etykietach zdarzyło się to w 14/72 przypadków. W tych 14 zadaniach odpowiedni minimaksowy środek obniżył najgorszy Brier średnio o 0,01974 względem środka punktowych skrajności. Największa poprawa wyniosła 0,05556. W przypadku skończonych kandydatów rozwiązujemy problem najmniejszej kuli obejmującej wszystkie wektory prognoz w normie ważonej rozkładem testowym. Zwracamy też rozkład na kandydatach dający dolną granicę ryzyka. Największa różnica między granicą górną i dolną wyniosła 9,88 × 10⁻¹⁰. [Rozwiązanie z dolną i górną granicą](https://github.com/damiankucharski/OKAPI/blob/f850e5f/okapi/brier_center.py)

W jawnej kontroli trzy równie ważne narożniki miały etykiety obserwowane 000. Dopuszczalne prawdy przy maksymalnie jednym błędzie to 000, 100, 010 i 001. Predykcja 1/3 w każdym punkcie osiąga dokładny najgorszy Brier 2/9. Punktowe skrajności dawały po uśrednieniu 1/2 w każdym punkcie i Brier 1/4. Przykład pokazuje też koszt dodatkowego założenia o liczbie błędów: gdy jest ono poprawne, gwarancja dotyczy prawdy. Przy nieznanym poziomie błędów wybór budżetu wymaga osobnych danych lub argumentu.

```sh
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.noisy_monotone --output /tmp/noisy-monotone
```

[Protokół](protocol.md) · [Wszystkie dopasowania](results/cells.csv) · [Dokładna kontrola](results/control.json) · [Czas](results/timing.json)
