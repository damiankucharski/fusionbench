# Trzy ukryte osie z surowych przykładów i etykiety większości

Dla trzech ortogonalnych separatorów w standardowym rozkładzie Gaussa można **odzyskać ich kierunki, włącznie ze zwrotem, z dokładnością do permutacji**. Wystarczają pierwszy i trzeci moment danych ważonych znakiem końcowej etykiety. Estymator nie potrzebuje prognoz zespołu, lokalnych etykiet, masek cech ani iteracyjnego uczenia. Nauczona funkcja to mediana trzech projekcji surowego wejścia. [Wyprowadzenie i założenia](derivation.md)

Pierwszy moment wskazuje sumę trzech osi. Spłaszczenie tensora trzeciego momentu Hermite’a wskazuje podprzestrzeń rozpiętą przez osie. W jej płaszczyźnie prostopadłej do sumy istotna składowa trzeciej harmonicznej ma trzy minima odległe o 120°. Wyznaczają one indywidualne osie. Test numeryczny na dokładnych momentach po dowolnym badanym obrocie odzyskał dokładne kierunki. Odwrócenie etykiet odwróciło orientacje.

### Ile przykładów potrzeba w dziewięciu wymiarach?

Dwanaście niezależnych zestawów osi, treningów i testów dało następujące średnie. Każdy test obejmował 4096 wierszy. Po zmianie wzmocnienia jeden losowo wybrany sygnał jest czterokrotnie silniejszy, dwa pozostałe dwukrotnie słabsze, a znaki i etykiety pozostają identyczne.

| Etykiety | Trafność IID | Trafność po zmianie | Kontrola obrotu po zmianie | Zgodność osi, cosinus | Pokrycie prawdziwej podprzestrzeni |
|---:|---:|---:|---:|---:|---:|
| 128 | 77,59% | 74,95% | 74,17% | 0,697 | 0,590 |
| 512 | 86,82% | 81,72% | 76,10% | 0,939 | 0,892 |
| 2048 | 93,73% | 90,17% | 77,05% | 0,989 | 0,980 |
| 8192 | 97,15% | 95,19% | 77,41% | 0,998 | 0,996 |

Kontrola obrotu zachowuje estymowaną sumę kierunków i dokładnie tę samą płaszczyznę. Tabela pokazuje średnią trafność z 12 równomiernie rozmieszczonych orientacji. Przy 2048 przykładach wybór orientacji na podstawie trzeciego momentu poprawił trafność po zmianie o **13,13 punktu procentowego**, ze sparowanym 95% przedziałem t od 12,21 do 14,04 pp. Cosinus wzrósł o 0,11433 (0,11375–0,11492). Po przetasowaniu końcowych etykiet trafność IID przy tej liczbie przykładów wyniosła 50,03%. Przy 128 przykładach prostszy klasyfikator liniowy oparty na pierwszym momencie miał wyższą trafność IID niż pełna rekonstrukcja, 80,23% i 77,59%.

### Rekonstrukcja bezpośrednia a inicjalizacja dalszego treningu

Po 600 krokach gładkiego treningu na 512 etykietach sieć startująca z estymowanych osi osiągnęła 96,01% IID i 91,35% po zmianie. Start z losowych ortonormalnych osi dał 95,19% i 88,98%. Start z losowej orientacji w estymowanej płaszczyźnie dał 95,94% i 90,99%. Podobnie przy 2048 przykładach przewaga trafności po dostrojeniu względem losowej orientacji w estymowanej płaszczyźnie miała przedział obejmujący zero. Dysponując podprzestrzenią, gradient często potrafił sam poprawić fazę.

Wynik o identyfikowalności dotyczy jawnego modelu większości, ortonormalnych ukrytych osi, zerowych progów i rozkładu N(0,I). Tabela mierzy odzyskanie kierunków i klasyfikacji w tym modelu. Skala logitów wszystkich bezpośrednich klasyfikatorów była ustalona na 5, dlatego Brier zależy również od tej konwencji. Kolejny test sprawdzi predykcję na rzeczywistych etykietach, bez założenia, że znamy ich regułę generatywną.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.majority_moments --output /tmp/majority-moments
```

[Zamrożony protokół](protocol.md) · [Metryki](results/metrics.csv) · [Kierunki](results/geometry.csv) · [Przedziały](results/paired.csv) · [Czas](results/timing.json)
