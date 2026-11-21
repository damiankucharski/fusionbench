# Co ujawnia etykieta większości, gdy przeczy jednemu głosowi?

Jeśli znak jednej prawdziwej osi wskazuje klasę przeciwną do większości trzech, **oba pozostałe głosy muszą zgadzać się z etykietą większości**. W kontrolowanym problemie użyliśmy tego faktu do odzyskania dwóch nieobserwowanych kierunków. Żaden przykład nie zawierał etykiety pojedynczego głosu. Silny kierunek wariancji surowych danych dostarczał kandydatury na pierwszą oś. Przy niezależnych gaussowskich resztach średnia znakowana etykietą w sprzecznych przykładach określa sumę dwóch pozostałych osi. Ich różnicę wyznacza najmniejsza warunkowa wariancja prostopadła do sumy. [Wyprowadzenie](derivation.md)

### Czy traktować kierunek dużej wariancji jako głos?

Przy stosunku dwóch największych wartości własnych mniejszym od 2 używaliśmy estymatora kątowego. Powyżej tej granicy orientowaliśmy pierwszą składową dodatnio względem etykiet. Zgodność co najmniej 62,5% uruchamiała metodę sprzeczności, a niższa prowadziła do usunięcia kierunku i odzyskania osi w reszcie przestrzeni. Progi były ustalone przed doświadczeniem. Wszystkie warianty widziały te same przykłady i końcowe etykiety.

| Rodzaj dużej wariancji | Wybór automatyczny | Trafność automatyczna | Zawsze potraktuj jako głos | Zawsze usuń | Zawsze użyj estymatora kątowego | Gładko uczona sieć |
|---|---|---:|---:|---:|---:|---:|
| Brak | Kątowy | 91,64% | 74,38% | 80,38% | 91,64% | 91,46% |
| Prawdziwy głos | Sprzeczności | **94,36%** | 94,36% | 72,03% | 73,91% | 86,80% |
| Czyste zakłócenie | Usuń | **92,22%** | 71,98% | 92,22% | 76,46% | 87,42% |
| Połowa głosu, połowa zakłócenia w sensie kwadratu projekcji | Sprzeczności | 82,66% | 82,66% | 72,10% | 70,70% | **85,90%** |

Wyniki to średnia trafność 16 niezależnych modeli uczonych na 2048 przykładach po teście zmieniającym względne wzmocnienia trzech sygnałów bez zmiany ich znaków. W każdym wierszu decyzja automatyczna była identyczna we wszystkich 16 powtórzeniach przy obu budżetach (512 i 2048 etykiet).

W wariancie z prawdziwym głosem około 510 z 2048 etykiet przeczyło kierunkowi największej wariancji. Automatyczny model osiągnął cosinus 0,99665 z prawdziwymi osiami i poprawił trafność po zmianie względem metody kątowej o 20,45 pp (sparowany 95% przedział t 19,64–21,26 pp). Przy czystym zakłóceniu jego usunięcie dało przewagę 20,24 pp nad traktowaniem go jako głosu (18,99–21,49 pp).

**Mieszany kierunek ujawnił ograniczenie.** Dla osi odchylonej o 45° od prawdziwego głosu zgodność jej znaku z etykietą treningową wyniosła 74,24%, blisko 75,11% obserwowanych dla osi prawdziwej. Sieć uczona gradientowo uzyskała wyższą trafność po zmianie o 3,24 pp (1,81–4,67 pp) i cosinus 0,9544 wobec 0,8974 metody sprzeczności. Sama zgodność z etykietami nie zapewniła poprawnej orientacji po zmianie rozkładu.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.majority_anchor --output /tmp/majority-anchor
```

[Protokół](protocol.md) · [Wszystkie wyniki](results/metrics.csv) · [Wybór ścieżki i liczba sprzeczności](results/routes.csv) · [Przedziały](results/paired.csv) · [Czas](results/timing.json)
