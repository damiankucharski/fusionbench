# Od drzewa OKAPI do jednej sieci ReLU

Badanie obejmuje osiem niezależnych problemów większości trzech sygnałów. Każdy sygnał jest znakiem projekcji innego trójwymiarowego bloku. Na 512 przykładach z lokalnymi etykietami nauczyliśmy trzy regresje logistyczne i dodaliśmy dokładną kopię pierwszej. Na osobnych 512 etykietach zadania OKAPI szukało drzewa min/max. Każde drzewo oceniliśmy na kolejnych 4096 przykładach. Losowania 800–807 i porównania zostały [zapisane przed obliczeniami](protocol.md).

| Model | Trafność | Brier binarny ↓ |
|---|---:|---:|
| Drzewo OKAPI z lokalnymi ekspertami | 84,86% | 0,14661 |
| Jedna sieć ReLU po kompilacji drzewa | 84,86% | 0,14661 |
| Znana mediana prognoz lokalnych ekspertów | 98,45% | 0,03434 |
| Topologia wybrana przez GP, nowe wagi uczone na etykietach zadania | 85,12% | 0,10629 |
| Znana topologia mediany, nowe wagi uczone na etykietach zadania | 86,85% | 0,09524 |
| MLP z ośmioma jednostkami ukrytymi, etykiety zadania | 82,39% | 0,12799 |
| Regresja logistyczna na pełnym wejściu, etykiety zadania | 83,37% | 0,11748 |

**Kompilacja zachowała funkcję wszystkich ośmiu drzew.** Największa bezwzględna różnica prawdopodobieństwa wyniosła 5,56 × 10⁻¹⁶. Monotoniczność sigmoidy pozwala obliczać min/max w przestrzeni logitów. Każda bramka wymaga jednego ReLU i dodawania lub odejmowania. Wyrażenia można współdzielić, ponieważ po progowaniu każdy min działa jak AND, każdy max jak OR. Równość tabel prawdy dla wszystkich binarnych wejść daje równość wartości dla dowolnych rzeczywistych logitów. Metoda wymaga wspólnych współrzędnych wejściowych, afinicznych logitów, binarnego zdarzenia i identycznościowego postprocessingu w źródłowym drzewie. Implementacja sprawdza pełne tabele domyślnie dla najwyżej 12 różnych map, a powyżej korzysta z identyczności składniowych.

**Większość naturalnych bramek pozostała.** Osiem źródłowych programów miało łącznie 17 bramek binarnych. Po kompilacji pozostało 16, po 0–5 na sieć z 1–3 aktywnymi projekcjami. Gdy celowo dopisaliśmy do każdego programu wyrażenie max(t,min(t,u)), kompilator odzyskał dokładnie liczbę bramek oryginalnego skompilowanego programu we wszystkich ośmiu przypadkach. Ten eksperyment mierzy liczbę operacji symbolicznych i zgodność numeryczną. Nie mierzy opóźnienia ani zużycia pamięci podczas wdrożenia.

**Wyszukiwanie struktury ograniczało jakość.** Dwa z ośmiu drzew poprawnie realizowały większość na wszystkich ośmiu narożnikach. Pozostałe osiągały 6/8 albo 7/8. Nawet prawidłowy wybór reguły nie rozwiązuje uczenia projekcji od zera. Wariant z topologią GP miał średni Brier o 0,02169 niższy od MLP (sparowany 95% przedział t dla poprawy 0,00436–0,03903). Przedziały dla różnic względem regresji liniowej i znanej topologii mediany obejmują zero. Topologia GP korzysta pośrednio z ekspertów w fazie odkrywania struktury. Nowe wagi trenowano na surowych danych i etykietach zadania, bez logitów nauczycieli. Wszystkie wyniki dotyczą jednego syntetycznego rozkładu i małego budżetu wyszukiwania.

Do odtworzenia potrzebna jest badawcza wersja OKAPI z modułem `okapi.lattice`, dostępna w katalogu sąsiednim lub zainstalowana w środowisku Pythona.

```sh
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.lattice_compile --output /tmp/lattice-compile
```

[Surowe metryki](results/metrics.csv) · [Wybrane programy i bramki](results/programs.csv) · [Sparowane przedziały](results/paired.csv) · [Czas pomiaru](results/timing.json)
