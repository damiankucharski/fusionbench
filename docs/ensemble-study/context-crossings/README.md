# Rozpoznawanie mocniejszego eksperta na niewidzianych kombinacjach jakości

Dwa modele potrafią oddawać równie często zgodne głosy w dwóch reżimach, w których decyzję przy konflikcie powinien podejmować inny ekspert. W tym eksperymencie rolę rozstrzygającego sygnału pełnił trzeci czujnik, dostępny wyłącznie w 32 osobnych, nieetykietowanych obserwacjach. Jedna sieć trenowana od zera wykorzystała go na kombinacjach jakości pominiętych w treningu. W krótkim treningu pomagało współdzielenie wag między czujnikami. Czterokrotne przejście zwykłej sieci po tych samych danych pozwoliło odzyskać większość tej przewagi.

## Projekt

Etykieta Y i trzy surowe czujniki mają wartości −1 lub +1. Każdy czujnik podaje prawidłowy znak z prawdopodobieństwem (1+r)/2, z szumem niezależnym warunkowo względem Y. Etykiety są zbalansowane. Siły r₁,r₂ zmieniają się niezależnie między partiami. Siła czujnika referencyjnego wynosi 0,75. Klasyfikowany przykład udostępnia tylko pierwsze dwa pomiary. Zbiór kontekstowy zawiera osobne przykłady i wszystkie trzy pomiary. Uczenie sieci korzysta wyłącznie z etykiet klasyfikowanych przykładów.

Trening losuje siedem komórek z siatki {0,25, 0,55, 0,85}². Obie kombinacje (0,25, 0,85) i (0,85, 0,25) są zarezerwowane do testu. Nie uczestniczą w doborze wag, kalibracji ani wyborze checkpointu. Podstawowym wynikiem jest Brier uśredniony po obu komórkach. Zaplanowane porównania obejmują 32 obserwacje względem zera, pełną i usuniętą informację referencyjną oraz jedną sieć i ensemble. Po dwóch seedach pilotażowych użyliśmy ośmiu świeżych seedów, po 256 zbiorów kontekstowych na komórkę. Ryzyko klasyfikacji dla każdego kontekstu oblicza się dokładnie po ośmiu możliwych trójkach (Y,X₁,X₂). Prawdziwe r wchodzą do tego obliczenia po zamrożeniu predykcji. Predyktor ze znanym reżimem ma jawny dostęp do r.

Sieci uczą się przewidywać dwa współczynniki głosowania z uśrednionych reprezentacji surowych trójek. Zastosowaliśmy addytywną postać logitu dla klasyfikowanego przykładu. W jednej architekturze oba współczynniki pochodzą z jednego odczytu. W wariancie ze współdzieleniem ta sama funkcja wylicza wagę każdego czujnika, zamieniając ich kolejność. Te warianty mają odpowiednio 348 i 335 parametrów. Wariant ze współdzieleniem oblicza enkoder dwukrotnie. Ensemble uśrednia prawdopodobieństwa trzech zwykłych sieci o różnych inicjalizacjach. Każda otrzymuje identyczny zbiór uczący. Drugi baseline uczy dwóch indywidualnych predyktorów binarnych na etykietach źródłowych i wybiera tego, który częściej zgadza się z referencją w zbiorze kontekstowym. Przy remisie uśrednia ich prawdopodobieństwa.

## Wynik potwierdzający

| Predyktor | Brier | Trafność |
|---|---:|---:|
| Jedna sieć ze współdzieleniem, bez kontekstu | 0,1262 | 77,50% |
| Jedna zwykła sieć, 32 obserwacje | 0,1258 | 80,01% |
| Ensemble trzech zwykłych sieci, 32 obserwacje | 0,1255 | 80,33% |
| Jedna sieć ze współdzieleniem, 32 obserwacje | 0,0966 | 91,93% |
| Wybór eksperta według zgodności z referencją | 0,0941 | 92,08% |
| Znany reżim, predyktor Bayesa | 0,0681 | 92,50% |

Różnica Brier między siecią ze współdzieleniem i ensemble’em trzech zwykłych sieci wyniosła −0,02895 [−0,03430, −0,02359]. W porównaniu z wyborem eksperta według zgodności wyniosła +0,00250 [−0,00204, +0,00704]. Nawiasy podają 95-procentowe przedziały t po ośmiu sparowanych seedach. Wariant ze współdzieleniem osiągnął 87,46% trafności z ośmioma obserwacjami i 92,50% ze 128 obserwacjami. W ostatnim przypadku Brier wynosił 0,0939, więc pozostała różnica w jakości samych prawdopodobieństw.

Zastąpiliśmy każdą referencję w kontekście niezależnym, zbalansowanym bitem i powtórzyliśmy trening. Rozkład dwóch głównych obserwacji ma wtedy postać P(X₁=x₁,X₂=x₂)=(1+r₁r₂x₁x₂)/4. Oba zastrzeżone reżimy mają identyczny iloczyn r₁r₂. Dowolna liczba takich nieetykietowanych trójek ma więc ten sam rozkład w obu przypadkach. Przy równoprawdopodobnych reżimach maksymalna trafność wynosi 77,5%. Sieć ze współdzieleniem osiągnęła 77,49%. Przy sprawnej referencji dwie zależności E[X₁H]=0,75r₁ i E[X₂H]=0,75r₂ rozróżniają reżimy.

## Czy potrzebna jest specjalna architektura?

Porównaliśmy jeden i cztery przebiegi po identycznych 360 minibatchach u pierwszych czterech seedów potwierdzających. Ta próba zachowuje inicjalizacje, przykłady, wszystkie etykiety i zbiory testowe. Zmienia liczbę aktualizacji. Każdy przebieg obejmuje 207 360 etykiet zadania w 8 640 partiach źródłowych. W 20% partii treningowych maska odcina całą informację kontekstową, co daje punkt odniesienia dla testu bez kontekstu.

| Predyktor | Trafność po 1 przebiegu | Trafność po 4 przebiegach | Brier po 4 przebiegach |
|---|---:|---:|---:|
| Zwykła pojedyncza sieć | 79,45% | 91,59% | 0,0937 |
| Ensemble trzech sieci | 78,75% | 91,43% | 0,0949 |
| Sieć ze współdzieleniem | 91,77% | 92,01% | 0,0894 |

Brier pojedynczej zwykłej sieci zmniejszył się o 0,0315 [0,0288, 0,0341]. Różnica z krótkiego treningu zależała w dużej mierze od budżetu optymalizacji. Prosty wybór eksperta osiąga podobną trafność przy niższym koszcie. Wynik dotyczy dwóch wymiennych, warunkowo niezależnych sensorów o dodatnich kompetencjach i stabilnej, poprawnie sparowanej referencji. Następne pytanie dotyczy tego, czy da się wykryć, że sama referencja zaczyna powtarzać błędy jednego eksperta.

## Reprodukcja

W głównym katalogu FusionBench, z PyTorch i zależnościami projektu:

```bash
python -m fusionbench.eval.context_crossings --output docs/ensemble-study/context-crossings/pilot --seeds 10 11
python -m fusionbench.eval.context_crossings --output docs/ensemble-study/context-crossings/confirmation --seeds 210 211 212 213 214 215 216 217
python -m fusionbench.eval.context_crossings --output docs/ensemble-study/context-crossings/replay --seeds 210 211 212 213 --steps 1440 --unique-steps 360 --paired-only
python docs/ensemble-study/context-crossings/compare_replay.py
python -m pytest -q tests/test_context_crossings.py
```

Każda seria zawiera konfigurację, tabelę wyników dla każdej komórki i seeda, średnie i sparowane kontrasty. Kolumna `partition=source` opisuje nowe, niezależnie wygenerowane konteksty dla reżimów znanych z treningu.
