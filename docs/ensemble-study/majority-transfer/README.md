# Test rzeczywistych etykiet po odkryciu osi większości

**Sieć większości z inicjalizacją momentami przegrała z pełnym zespołem we wszystkich 19 ocenianych jednostkach.** Było to pięć foldów Wisconsin, pięć foldów parzystości cyfr oraz dziewięciu niewidzianych uczestników rozpoznawania chodzenia w górę lub w dół po schodach. Ta granica dotyczy konkretnej architektury trzech projekcji i treningu przez dziewięć składowych PCA. Matematyczne odzyskiwanie trzech osi było wcześniej uzasadnione dla konkretnego generatora gaussowskiego. Tutaj mierzymy wyłącznie predykcję na rzeczywistych etykietach.

| Model, Brier ↓ | Wisconsin | Parzystość cyfr | HAR, kierunek na schodach |
|---|---:|---:|---:|
| Bezpośrednia rekonstrukcja momentami | 0,07754 | 0,14644 | 0,10940 |
| Sieć większości, start z momentów | 0,03184 | 0,08040 | 0,07105 |
| Sieć większości, start z losowych osi | 0,03437 | 0,08497 | 0,07373 |
| MLP, trzy jednostki ReLU na tym samym PCA | 0,02134 | 0,08372 | 0,08829 |
| Logistyczny na tym samym PCA | 0,02525 | 0,12449 | 0,07367 |
| RBF SVC na tym samym PCA | 0,03173 | 0,03260 | 0,06106 |
| Logistyczny na pełnych cechach | 0,02130 | 0,06526 | **0,01289** |
| RBF SVC na pełnych cechach | 0,02223 | **0,01223** | 0,02138 |
| Las losowy na pełnych cechach | 0,03174 | 0,02967 | 0,05942 |
| Średnia trzech poprzednich modeli | 0,02131 | 0,02408 | 0,02198 |

Sieć inicjalizowana momentami uzyskała odpowiednio 97,01%, 88,76% i 89,89% trafności. Zespół osiągnął 97,36%, 97,72% i 97,93%. Poprawa średniego Brier względem losowej inicjalizacji wyniosła 0,00253, 0,00457 i 0,00268. Wśród pojedynczych modeli z pełnym wejściem regresja logistyczna osiągnęła najniższy Brier na Wisconsin i HAR, a RBF SVC na cyfrach. Takie zestawienie najlepszych wyników jest opisowe, ponieważ wyboru między rodzinami nie przeprowadzaliśmy na osobnej walidacji.

PCA i standaryzację dopasowywaliśmy wyłącznie na treningu. Dziewięć składowych zachowywało około 94% wariancji Wisconsin, 56–57% cyfr i 61% HAR. Silny RBF w tych samych dziewięciu wymiarach również wyprzedził sieć większości w średnim Brier na cyfrach i HAR. Wszystkie trzy liście sieci oraz oba przekształcenia liniowe złożyliśmy w jedną mapę afiniczną z surowego wejścia. Po tej zmianie maksymalny błąd prawdopodobieństwa wyniósł 5,67 × 10⁻¹⁵. Sieci w tej postaci mają 93, 195 i 1686 współczynników, a wyjście to sigmoida mediany trzech liczb.

W Wisconsin i na cyfrach każdy wiersz trafił do testu dokładnie raz. Treningi foldów częściowo się pokrywają. Dla cyfr brakuje identyfikatorów autorów. Na HAR wykorzystaliśmy oficjalny podział na osoby, 2059 okien treningowych od 21 osób oraz 891 testowych od dziewięciu. W metryce HAR każda osoba ma jednakową wagę. Próby nie dostarczają pomiaru niezależnego powtarzania badań na nowych populacjach. Kolejne porównanie powinno wybrać rodzinę pojedynczego modelu na danych treningowych i skonfrontować ją z zespołem na nowych zadaniach.

```sh
curl -fL 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip' -o har.zip
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.majority_transfer --har-archive har.zip --output /tmp/majority-transfer
```

[Protokół](protocol.md) · [Prognozy poza treningiem](results/predictions.csv.gz) · [Wyniki w foldach i osobach](results/units.csv) · [Weryfikacja przekształcenia na surowe wejście](results/splits.csv) · [Czas](results/timing.json)
