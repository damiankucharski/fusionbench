# Kiedy jedna sieć uczy się większości trzech ekspertów?

Na **12 nowych niezależnych problemach** przeszukanie 18 różnych reguł min/max wskazało medianę trzech prognoz w każdym przypadku. Jedna sieć trenowana wyłącznie na surowych wejściach i końcowych etykietach osiągnęła **97,96% trafności**, gdy znała regułę większości i prawidłowy podział dziewięciu cech na trzy bloki. Każde doświadczenie miało osobny test liczący 4096 przykładów. Warunki i porównania ustaliliśmy [przed obliczeniami](protocol.md).

### Odkrywanie reguły z prognoz lokalnych ekspertów

Każdy z trzech ekspertów nauczył się znaku lokalnej projekcji z osobnych etykiet pojęć, na wspólnych 512 przykładach źródłowych. Na kolejnych 512 przykładach z etykietą końcową szukaliśmy reguły fuzji. W klasie wszystkich niebędących stałą, monotonicznych funkcji trzech wejść istnieje 18 różnych tabel prawdy. Każdą można przedstawić jako maksimum z minimów, wyliczyć dla rzeczywistych prawdopodobieństw i ocenić przez empiryczny Brier.

| Reguła | Poprawne wszystkie narożniki | Testowy Brier ↓ | Trafność |
|---|---:|---:|---:|
| Przegląd 18 funkcji | 12/12 | 0,03432 | 98,61% |
| Znana mediana | 12/12 z definicji | 0,03432 | 98,61% |
| GP min/max, 30 pokoleń | 0/12 | 0,18298 | 80,79% |

GP miało także czwartą kolumnę, dokładną kopię pierwszej, oraz domyślne pomocnicze kryterium minimalizacji liczby węzłów. Przegląd korzystał z trzech różnych identyfikatorów, co usuwa równoważne programy. W obu przypadkach klasa reprezentowalnych funkcji na tych wejściach jest ta sama. Średnia różnica Brier na korzyść pełnego przeglądu wyniosła 0,14865 (sparowany 95% przedział t 0,11548–0,18182). Ograniczenie implementacji do maksymalnie czterech wejść wynika z szybkiego wzrostu klasy funkcji. Dla czterech jest ich 166.

### Uczenie jednej sieci bez prognoz ekspertów

Każda sieć oblicza medianę trzech uczonych projekcji afinicznych. Model z trzema blokami ma 12 czynnych współczynników razem z wyrazami wolnymi, model gęsty ma ich 30. Przy błędnym podziale każda projekcja widzi po jednej współrzędnej każdego prawdziwego sygnału. Obrót ortogonalny zachowuje całe wejście oraz etykiety i niszczy zgodność dawnych bloków współrzędnych z sygnałami. Wyniki poniżej to średnie Brier, im mniej tym lepiej.

| Etykiety zadania i geometria | Pierwotne bloki | Skrzyżowane bloki | Gęste projekcje | Regresja logistyczna |
|---|---:|---:|---:|---:|
| 128, zgodne osie | 0,06168 | 0,18183 | 0,14711 | 0,13409 |
| 512, zgodne osie | **0,04367** | 0,15522 | 0,08713 | 0,11780 |
| 128, po obrocie | 0,19270 | 0,18307 | 0,14224 | **0,13398** |
| 512, po obrocie | 0,15987 | 0,16039 | **0,08403** | 0,11777 |

Przy 512 etykietach i zgodnych osiach poprawny podział zmniejszył Brier o 0,11155 względem skrzyżowanego (95% 0,09946–0,12363). Po obrocie użycie dawnych bloków pogorszyło Brier o 0,07584 względem sieci gęstej (95% 0,05189–0,09978). Gęsta sieć uzyskała wtedy 88,72% trafności wobec 76,56% przy dawnych maskach. Architekturę większości wskazaliśmy wszystkim sieciom z trzema projekcjami. Osiągnięcie 97,96% wymagało ponadto poprawnej informacji o grupach cech. Eksperyment nie mierzy rozpoznania architektury i grup cech jednocześnie na danych rzeczywistych.

```sh
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.lattice_views --output /tmp/lattice-views
```

[Implementacja dokładnego przeglądu w OKAPI](https://github.com/damiankucharski/OKAPI/blob/c614670/okapi/lattice_search.py) · [Metryki](results/metrics.csv) · [Programy](results/programs.csv) · [Przedziały](results/paired.csv) · [Czas](results/timing.json)
