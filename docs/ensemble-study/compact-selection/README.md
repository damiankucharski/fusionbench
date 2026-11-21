# Wybór jednego modelu przed ujawnieniem etykiet testu

Na dwóch z trzech rzeczywistych zadań walidacja wybrała jedną sieć radialną z 64 centrami. Na każdym z dziesięciu zewnętrznych foldów tych dwóch zbiorów jej Brier był niższy niż dla średniej regresji logistycznej, RBF SVC i lasu 150 drzew. Na Spambase ta sama reguła pozostawiła zespół we wszystkich pięciu foldach. Pełny protokół i tolerancja 0,005 były [zapisane przed treningiem](protocol.md).

| Zbiór UCI | Wiersze | Wybrany kompakt, Brier ↓ | Zespół, Brier ↓ | Polityka z walidacji | Trafność kompaktu / zespołu |
|---|---:|---:|---:|---|---:|
| Banknote Authentication | 1372 | 0,00045 | 0,00340 | 5 × sieć radialna | 100,00% / 100,00% |
| Ionosphere | 351 | 0,03313 | 0,04659 | 5 × sieć radialna | 95,44% / 94,59% |
| Spambase | 4601 | 0,06790 | 0,04764 | 5 × zespół | 91,15% / 93,78% |

Na każdym z pięciu zewnętrznych foldów trzy wewnętrzne foldy wskazywały lepszy kompakt spośród regresji logistycznej i sieci radialnej. Tę samą wewnętrzną ocenę porównywaliśmy z zespołem, dopuszczając wzrost Brier o 0,005. Wybrany sposób wdrożenia ponownie trenowaliśmy na całym treningu zewnętrznym. Każda etykieta testowa służyła ocenie raz. Identyczne surowe wektory pozostawały w tej samej grupie w zewnętrznych foldach, wewnętrznej selekcji i kalibracji SVC.

Decyzje z walidacji zgodziły się z późniejszym statusem tolerancji w 15/15 foldach. Znak różnicy był jednakowy wewnątrz każdego zbioru. Podziały tego samego zadania mają wspólne przykłady treningowe, więc to trzy zbadane zadania. Na Spambase kompakt wybrany między dwiema rodzinami wypadł gorzej niż stałe użycie sieci 64-centrowej (0,06790 i 0,06422). Sam las uzyskał 0,04322, całe uśrednienie 0,04764. Na Banknote sam RBF SVC uzyskał 0,00034. Tabela odpowiada na pytanie o zastąpienie konkretnego zespołu przy tym budżecie.

### Co trzeba wdrożyć

Centra Nyströma to 64 przykłady wybrane z treningu. Współczynnik Gaussa wynosił 1/d na wystandaryzowanych wejściach, a liniowy odczyt uczono wyłącznie na oryginalnych etykietach. Macierz normalizującą i odczyt złożyliśmy w jeden skalar na centrum oraz wyraz wolny. Predykcja wymaga 64 odległości, 64 funkcji Gaussa i jednej sigmoidy. Pełny zespół obejmował m.in. SVC z 84–110 wektorami nośnymi na Banknote/Ionosphere oraz 1057–1083 na Spambase, a także lasy z odpowiednio 5708–7434 i 85 276–87 572 węzłami. Są to liczby składowych, bez pomiaru opóźnienia wdrożenia.

Wśród testów zewnętrznych największa różnica prawdopodobieństwa między skompilowaną i oryginalną siecią wyniosła 3,22 × 10⁻¹⁰. We wszystkich dopasowaniach walidacyjnych maksimum wyniosło 1,10 × 10⁻⁹. Pierwsze uruchomienie zakończyło się na wewnętrznej kontroli numerycznej 10⁻⁹. Po zapisaniu każdej faktycznej różnicy zamiast przerywania powtórzyliśmy cały zamrożony protokół z tymi samymi hiperparametrami.

```sh
python -m examples.uci_three_binary --download /tmp/uci3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.compact_selection --data-dir /tmp/uci3 --output /tmp/compact-selection
```

[Wybory i koszty według foldów](results/selection.csv) · [Predykcje poza treningiem](results/predictions.csv.gz) · [Metryki członków zespołu](results/means.csv) · [Dane i ich sumy SHA-256](../../../examples/uci_three_binary.py) · [Czas udanej i przerwanej próby](results/timing.json)
