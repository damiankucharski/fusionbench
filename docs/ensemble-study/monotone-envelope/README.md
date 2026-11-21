# Kiedy wszystkie zgodne programy OKAPI zastępują dwie reguły skrajne?

Dla pełnej klasy monotonicznych, niestałych funkcji binarnych **dwie reguły wystarczają do dokładnej predykcji minimaksowej w sensie Brier**. Wyznaczamy je bez wyszukiwania GP i bez enumeracji reguł. Dolna skrajność daje 1 wszędzie powyżej zaobserwowanego przypadku dodatniego. Górna daje 0 wszędzie poniżej zaobserwowanego przypadku ujemnego. Porządek jest współrzędny, a przypadki samych zer i samych jedynek mają znane etykiety 0 i 1. Kod zachowuje tylko minimalne obserwacje dodatnie i maksymalne ujemne. [Implementacja OKAPI](https://github.com/damiankucharski/OKAPI/blob/9b72f4e/okapi/monotone_envelope.py)

Obie skrajności są dopuszczalnymi funkcjami zgodnymi ze wszystkimi bezszumowymi etykietami. Na każdym pozostałym punkcie albo dają tę samą odpowiedź, albo są przeciwne. Niech D oznacza masę drugiej grupy pod dowolnym testowym rozkładem. Średnia prognoz obu skrajności daje 0,5 na całej grupie spornej, więc ma Brier D/4 względem **każdej** możliwej prawdy. Żaden inny predyktor nie ma niższego najgorszego ryzyka, ponieważ na grupie spornej średnia strat względem dwóch przeciwnych skrajności jest co najmniej 1/4. Zbiór samych danych treningowych określa zatem jeden trójwartościowy predyktor. Wejścia stanowią binarne zdarzenia ekspertów, a trening wymaga końcowych etykiet zadania. Wersja ta nie rekonstruuje wag sieci generujących te zdarzenia.

### Sprawdzenie na wszystkich 166 regułach czterech wejść

Każdą niestałą funkcję rozważaliśmy kolejno jako prawdę. W każdej z 12 niezależnych prób losowaliśmy źródłowe bity z prawdopodobieństwem 0,2, a test przypisywał jednakową wagę wszystkim 16 narożnikom. Granice wyliczone z przykładów były zgodne z pełnym przeglądem i obie należały do rodziny we wszystkich 7968 dopasowaniach.

| Etykiety | Średnia liczba zgodnych reguł | Masa sporna | Najgorszy Brier skrajności | Najgorszy Brier średniej wszystkich zgodnych | Najgorszy Brier najprostszego zgodnego programu |
|---:|---:|---:|---:|---:|---:|
| 4 | 92,56 | 64,50% | 0,16125 | 0,21487 | 0,46392 |
| 12 | 31,71 | 39,07% | 0,09768 | 0,12657 | 0,34830 |
| 32 | 12,70 | 23,04% | 0,05759 | 0,06999 | 0,21533 |
| 64 | 4,15 | 10,75% | 0,02688 | 0,02977 | 0,10646 |

Średnia względem jednostajnego rozkładu prawdziwych reguł premiuje inne rozwiązanie. Przy 32 etykietach średnia wszystkich zgodnych programów uzyskała Brier 0,04908, podczas gdy predyktor minimaksowy osiągnął 0,05759. W każdej komórce zachowaliśmy zarówno średnie, jak i najgorsze ryzyko.

### Kiedy dwie skrajności przestają wystarczać?

W kontroli dopuszczalne są wyłącznie cztery pojedyncze wejścia, a trening obserwuje tylko przypadki samych zer i samych jedynek. Wszystkie cztery projekcje są zgodne. Ich punktowe minimum i maksimum to AND i OR, więc obie te funkcje leżą poza ograniczoną rodziną. Przy niezależnych, uczciwych bitach średnia wszystkich czterech głosów ma ten sam Brier 0,18750 względem każdego z nich. Jest optymalna, bo pod jednostajnym priorem ta średnia minimalizuje ryzyko, a najgorsze ryzyko jest co najmniej średnim. Środek punktowych skrajności osiąga **0,21875**, a najlepsza mieszanina najwyżej dwóch dopuszczalnych głosów **0,37500**. Dla dwóch wybranych głosów równy podział minimalizuje wariancję, ale pozostały, pominięty sygnał jest od obu niezależny. Sama monotoniczność każdego modelu nie gwarantuje, że rodzina zawiera swoje skrajności.

### Koszt dodatkowych wejść

W wyższym wymiarze prawdziwa reguła wykorzystywała tylko trzy ukryte spośród dostępnych bitów. Po 512 etykietach na ośmiu wejściach sporna masa testowa wynosiła 14,10%. Przy 16 wejściach wynosiła 48,36%. Algorytm wciąż wyliczał ścisłe granice bez enumeracji, lecz same dane nie odrzucały wystarczająco wielu innych monotonicznych wyjaśnień. Sprzeczne etykiety lub odwrócona zależność między wejściem i celem naruszają założenia, a implementacja sygnalizuje obserwowalny konflikt.

```sh
PYTHONPATH=.:../okapi OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.monotone_envelope --output /tmp/monotone-envelope
```

[Protokół](protocol.md) · [Wszystkie dopasowania](results/closed_cells.csv.gz) · [Ograniczona rodzina](results/restricted_means.csv) · [Kontrola czterech głosów](results/four_coordinate_control.json) · [Większe wymiary](results/higher_means.csv) · [Czas](results/timing.json)
