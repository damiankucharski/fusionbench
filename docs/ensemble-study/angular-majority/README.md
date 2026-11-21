# Odzyskiwanie trzech osi niezależnie od długości przykładów

Wynik o odzyskaniu trzech ukrytych separatorów obejmuje wejścia o **dowolnych dodatnich długościach**, jeżeli ich kierunki mają rozkład jednostajny na sferze. Promień może zależeć od kierunku. Etykieta musi być większością znaków trzech projekcji o zerowym progu na ortonormalne osie. Każdą obserwację treningową normalizujemy do długości √d. Sieć wdrożeniowa używa jednej mediany trzech projekcji surowego wektora. [Pierwotne wyprowadzenie dla rozkładu Gaussa](../majority-moments/derivation.md)

### Skąd poprawka d/(d+1)?

Dla wejścia gaussowskiego G = R S, kierunek S i promień R są niezależne. W dowolnym wymiarze d stosunek E[R³]/E[R] promienia chi wynosi d+1. Po ustaleniu H = √d S oraz m = E[Y H] tensor

```math
T_{\mathrm{ang}}=E[Y H^{\otimes3}]-\frac{d}{d+1}\operatorname{sym}(m\otimes I)
```

jest dodatnią wielokrotnością pierwotnego tensora gaussowskiego. Operator sym oznacza sumę trzech rozmieszczeń wektora w tensorze. Rozpiętość i trzecia harmoniczna odzyskują te same osie. Ponieważ dodatnie przeskalowanie każdego wiersza zachowuje H dla tego wiersza, również w **skończonej próbie** estymator jest niezmienniczy na takie zmiany długości treningowych. Predykcje porównujemy tutaj przy tych samych wejściach testowych. Wielkość logitu sieci nadal zależy od długości podanego jej wejścia testowego.

### Szesnaście niezależnych problemów

W poniższej tabeli każda sieć używa 2048 końcowych etykiet treningowych. Test po zmianie wzmocnienia zwiększa czterokrotnie tylko jeden z trzech sygnałów w wierszu i zmniejsza dwukrotnie oba pozostałe, zachowując całą trójkę znaków. Kolumny dotyczą rzeczywistej zgodności ukrytych kierunków i testowych prognoz.

| Trening | Estymator | Cosinus osi | Trafność IID | Trafność po zmianie |
|---|---|---:|---:|---:|
| Gaussowski | Gaussowski | 0,9900 | 93,92% | 90,15% |
| Gaussowski | Kątowy | 0,9939 | 95,13% | 91,46% |
| 2% wierszy o długości ×40 | Gaussowski | 0,5650 | 75,95% | 74,47% |
| 2% wierszy o długości ×40 | Kątowy | 0,9939 | 95,13% | 91,46% |
| Kierunki częstsze wzdłuż pierwszej osi | Gaussowski | 0,8953 | 82,29% | 67,63% |
| Kierunki częstsze wzdłuż pierwszej osi | Kątowy | 0,5823 | 76,14% | 73,66% |

Przy zakłóceniu długości kątowy estymator poprawił trafność po zmianie o 16,99 punktu procentowego (sparowany 95% przedział t 15,98–17,99 pp) oraz cosinus o 0,4289 (0,3747–0,4830). We wszystkich parach 512, 2048 i 8192 przykładów maksymalna zmiana prawdopodobieństwa po dodatnim przeskalowaniu tych samych wierszy treningowych wyniosła 8,66 × 10⁻¹⁵. Jednostkowy współczynnik w miejscu d/(d+1) również zachował osie w kontrolnym rachunku populacyjnym. Otrzymany tensor miał rangę dziewięć, a z poprawką rangę trzy.

### Kiedy normalizacja zawodzi?

W kontroli kierunkowej zastąpiliśmy wielkość pierwszej projekcji zmienną chi z siedmioma stopniami swobody. Znak i wszystkie pozostałe współrzędne zachowaliśmy. Taki rozkład kierunków waży jednostajną sferę proporcjonalnie do szóstej potęgi pierwszej współrzędnej. Wektor pierwszego momentu ma w prawdziwych osiach współczynniki proporcjonalne do (3,2; 1; 1), ponieważ E|Z|⁷ / (E[Z⁶] E|Z|) = 48/15 dla standardowego Z. Populacyjny cosinus z poprawną sumą osi wynosi 0,85813. Przy 8192 przykładach zaobserwowaliśmy 0,85582, a zgodność indywidualnych osi pozostała na poziomie 0,58183. Zwiększenie liczby przykładów nie usuwa tego obciążenia pierwszego momentu.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.angular_majority --output /tmp/angular-majority
```

[Protokół](protocol.md) · [Ranga i wartości osobliwe](results/population.json) · [Surowe metryki](results/metrics.csv) · [Geometria](results/geometry.csv) · [Niezmienniczość](results/invariance.csv) · [Czas](results/timing.json)
