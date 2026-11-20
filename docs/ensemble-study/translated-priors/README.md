# Co odróżnia przesunięcie instrumentu od zmiany proporcji klas?

Z samej średniej jednego wyniku nie można rozdzielić tych dwóch zmian. Pełny rozkład wyniku zawiera więcej informacji. W kontrolowanym doświadczeniu po przesunięciu wszystkich czujników estymator średniej zgłaszał 80,6% klasy dodatniej przy prawdziwych 50%. Dopasowanie dwóch źródłowych rozkładów warunkowych z jedną wspólną translacją dało **50,1%**. Na rzeczywistych pomiarach dziewięciu osób ta sama metoda pomogła przy sztucznej translacji, lecz zaszkodziła przy rzeczywistej zmianie proporcji klas bez translacji.

## Wynik z jednego kanału

Przechowujemy dwie empiryczne dystrybuanty źródłowe F₀ i F₁ wyniku eksperta, wyznaczone z etykiet. Szukamy udziału klasy π i przesunięcia d, dla których docelowa dystrybuanta jest najbliższa (1−π)F₀(x−d)+πF₁(x−d). Dla ustalonego d estymacja π jest jednowymiarową regresją liniową. OKAPI przeszukuje 301 wartości d. Wynikiem jest estymowany prior, przesunięcie w pierwotnych jednostkach i błąd dopasowania dystrybuanty. Wymagane są stabilne kształty obu rozkładów klas z dokładnością do wspólnego przesunięcia oraz wystarczająco odróżnialne rozkłady. W eksperymencie nie dostrajamy estymatora do etykiet celu.

W polarity_fault metoda korzysta z wyników jednego wskazanego czujnika, a następnie wykorzystuje estymowany prior do przeniesienia wag i przesunięcia pozostałego bloku. Wyniki poniżej to nadwyżka oczekiwanego Brier nad dokładnym posteriorem, na 12 odłożonych bankach i przy budżecie 1024 nieetykietowanych pomiarów. W podstawowym warunku redukcja względem estymacji z samej średniej wyniosła **0,11176 [0,10278; 0,12074]**. Przy zaledwie 128 przykładach nadwyżka wyniosła 0,00490 wobec 0,12276 z samej średniej.

| Warunek docelowy | Model zamrożony | Tylko średnia | Dystrybuanta bez translacji | Dystrybuanta z translacją |
|---|---:|---:|---:|---:|
| Bez zmian | 0,0003 | 0,0004 | 0,0003 | 0,0008 |
| Tylko zmiana klas | 0,0119 | 0,0004 | 0,0004 | 0,0007 |
| Przesunięcie wszystkich czujników | 0,0476 | 0,1127 | 0,0893 | 0,0009 |
| Przesunięcie i zmiana klas | 0,0078 | 0,0871 | 0,0854 | 0,0007 |
| Zmiana skali wszystkich czujników ×1,7 | 0,0059 | 0,0789 | 0,0789 | 0,0790 |

Zmiana skali narusza także założenie jednostkowej wariancji używane do przeniesienia wag. Błąd dopasowania dystrybuanty wzrósł do 0,1054 z 0,0071 bez zmian. Wyniki pokazują potencjał tej diagnostyki, ale nie kalibrowano progu alarmowego.

## Odrębne osoby, siedzenie i stanie w UCI HAR

Zbudowaliśmy regresję logistyczną na rzeczywistych pomiarach inercyjnych 13 osób. Ośmiu kolejnych uczestników dostarczyło etykiet źródłowych do kalibracji oraz dystrybuant. Każda z dziewięciu nowych osób daje 23–42 nieetykietowane przykłady do adaptacji. Test wykorzystuje późniejsze przykłady tej samej osoby, z przerwą co najmniej pięciu okien osobno dla każdej aktywności. Zmienialiśmy sztucznie udział stania i dodawaliśmy znane przesunięcie do wyniku modelu. Etykiety celu wykorzystał jedynie konstruktor warunków i ocena.

| Udział stania | Przesunięcie w źródłowych odchyleniach | Brier zamrożonego modelu | Brier adaptacji z samej średniej | Brier adaptacji z pełnej dystrybuanty |
|---:|---:|---:|---:|---:|
| 50% | 0 | 0,0857 | 0,1049 | 0,1144 |
| 50% | 0,6 | 0,2246 | 0,4171 | 0,1099 |
| 80% | 0 | 0,0457 | 0,1141 | 0,1899 |
| 80% | 0,6 | 0,0885 | 0,1734 | 0,1363 |

Przy samym przesunięciu Brier spadł względem zamrożenia o **0,1147 [0,0206; 0,2088]**, a względem samej średniej o 0,3072 [0,2274; 0,3871]. Przy 80% stania i zerowym przesunięciu estymator przypisał średnio 0,56 odchylenia przesunięciu i oszacował udział stania na 59%. Niewielki błąd dopasowania dystrybuanty (0,0327) współwystępował z dużym błędem modelu. W sytuacji zmienności pomiędzy ludźmi dobra zgodność jednego rozkładu marginalnego nie wystarczyła do wiarygodnego rozpoznania mechanizmu. Przedziały t traktują osoby jako jednostki analizy; przy dziewięciu osobach mają ograniczoną precyzję.

## Odtworzenie i kolejny test

Wymagany jest archiwalny zbiór UCI HAR w formacie wskazanym przez wcześniejsze badanie real-transfer. Skrypt sprawdza SHA-256 archiwum. Pliki CSV zawierają wszystkie wyniki, w tym oddzielny komparator znający parametry sztucznej interwencji. Całość zajęła 2,93 s CPU wall time, w tym 0,93 s na symulację. Nie trenowano ponownie drzew GP.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python fusionbench/examples/translated_priors.py --har har.zip --out translated-priors-results
```

Następna hipoteza dotyczy bogatszego świadka zespołowego. Trzy warunkowo niezależne kanały pozwalają oszacować prior z ich wspólnych momentów po wycentrowaniu, nawet gdy różne stałe przesunięcia dotknęły wszystkich urządzeń. Trzeba sprawdzić wariancję przy małych budżetach i błąd po naruszeniu niezależności.

[Przedrejestracja](https://github.com/damiankucharski/fusionbench/blob/e1e4322/docs/ensemble-study/translated-priors/protocol.md) · [Kod OKAPI](https://github.com/damiankucharski/OKAPI/blob/fbe160e/okapi/score_shift.py) · [UCI HAR, źródło danych](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
