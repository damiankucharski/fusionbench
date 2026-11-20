# Jak jedna sieć odzyskuje kierunki trzech ukrytych sygnałów

Na 12 niezależnych problemach zmiana funkcji użytej w treningu pozwoliła tej samej sieci lepiej odzyskać ukryte kierunki. Przy 512 etykietach końcowych zgodność z prawdziwymi osiami wzrosła z **0,798 do 0,952**, a trafność po zmianie amplitud zachowującej etykietę z **80,98% do 87,16%**. W obu przypadkach wdrożono tę samą medianę trzech logitów. Model miał 30 parametrów i uczył się na surowych, dziewięciowymiarowych wejściach. W obliczaniu jego gradientów brały udział wyłącznie etykiety zadania.

### Dwie funkcje treningowe dla tej samej sieci

Trzy uczone mapy afiniczne wytwarzają logity. Każdy prawdziwy sygnał jest znakiem nieznanej projekcji izotropowego rozkładu Gaussa, a końcowa etykieta to większość trzech znaków. Twarda mediana kieruje gradient do środkowego logitu, gdy logity są różne. Gładka wersja oblicza prawdopodobieństwo większości przy trzech niezależnych zdarzeniach Bernoulliego. Dla prawdopodobieństw p, q i r otrzymujemy pq + pr + qr − 2pqr. Przy skończonych logitach gradient dociera do wszystkich trzech wejść. Ich wagi inicjalizowano identycznie w porównywanych modelach, z różnymi wartościami pomiędzy liśćmi.

| Etykiety | Trening, potem dekoder medianowy | Trafność IID | Brier IID | Trafność po zmianie | Brier po zmianie | Cosinus osi |
|---:|---|---:|---:|---:|---:|---:|
| 512 | Mediana | 88,10% | 0,08796 | 80,98% | 0,15604 | 0,798 |
| 512 | Gładka większość | 93,05% | 0,06240 | 87,16% | 0,10303 | 0,952 |
| 512 | Gładka większość, trzy jednakowe starty | 83,17% | 0,12186 | 77,16% | 0,19141 | 0,571 |
| 2048 | Mediana | 94,64% | 0,05135 | 85,07% | 0,11468 | 0,964 |
| 2048 | Gładka większość | 96,60% | 0,04840 | 90,37% | 0,08098 | 0,994 |
| 2048 | Gładka większość, trzy jednakowe starty | 83,67% | 0,11900 | 77,36% | 0,19157 | 0,575 |

Cosinus osi to średnia podpisanych iloczynów skalarnych jednostkowych kierunków po optymalnym dopasowaniu permutacji. Znane osie służyły jedynie do generowania danych, interwencji i oceny. W osobnym teście jeden losowo wybrany sygnał każdego przykładu wzmocniliśmy czterokrotnie, dwa pozostałe osłabiliśmy dwukrotnie. Znaki wszystkich trzech i etykiety są wtedy dokładnie takie same. Porównywane modele widziały identyczne wiersze i przeszły ten sam harmonogram 600 aktualizacji.

Przy 512 etykietach sparowany 95% przedział t dla poprawy Brier po zmianie wynosił 0,01916–0,08686. Dla poprawy cosinusa wynosił 0,0111–0,2967. Przedział dla poprawy Brier na danych IID obejmował zero. Z gładkim dekoderem po gładkim treningu Brier po zmianie był wyższy niż z medianą przy tych samych nauczonych wagach (0,10924 oraz 0,10303). Regresja liniowa i zwykła sieć ReLU z trzema jednostkami uzyskały po zmianie odpowiednio 0,19869 i 0,19141. Przy identycznych początkowych wagach wszystkich trzech liści średni wzajemny cosinus nauczonych kierunków pozostał równy 1.

Architektura w obu wariantach kodowała prawidłową regułę większości. Kontrolowana zmiana amplitud sprawdza skutki różnego rozpoznania składowych funkcji w tym modelu generatywnym. Wynik eksperymentalny nie ustala zbieżności treningu dla dowolnych danych.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.majority_routing --output /tmp/majority-routing
```

[Zamrożony protokół](protocol.md) · [Metryki](results/metrics.csv) · [Geometria dla każdego seeda](results/geometry.csv) · [Przedziały](results/paired.csv) · [Czas](results/timing.json)
