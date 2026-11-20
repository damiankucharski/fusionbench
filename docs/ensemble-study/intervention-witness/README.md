# Czy podmiana wejścia ujawni zależną referencję bez etykiet?

W teście z dwoma obserwatorami cztery kontrolowane podmiany dostarczonych predykcji ujawniły bezpośrednio zależnego obserwatora w 98,46% partii. Wybór drzewa OKAPI z użyciem pozostałej referencji dał Brier 0,0953 wobec 0,1446 dla stałej mieszaniny. Różnica wyniosła −0,0493 [−0,0500, −0,0486] w 12 nowych sparowanych próbach. Gdy zależność obserwacyjna wynikała ze wspólnego wcześniejszego pomiaru, interwencja nie rozróżniała referencji.

## Dwa źródła tej samej zgodności

Użyliśmy równoprawdopodobnych reżimów, w których silniejszy jest inny z dwóch czujników. Jedna z dwóch referencji bezpośrednio lub pośrednio dzieli błędy słabszego czujnika. Jej identyfikator jest losowy i stały w partii. Wcześniejsza konstrukcja gwarantuje, że wszystkie 16 normalnie obserwowanych wzorców znaków może mieć identyczne prawdopodobieństwa w przeciwnych reżimach. Wybór źródłowych kandydatów korzystał z faktycznych przebiegów GP. Każdy z 24 przebiegów wybrał liść odpowiadający ekspertowi właściwemu dla swojego reżimu.

Referencja wykorzystuje rzeczywisty operator OKAPI wybierający predykcję najdalszą od progu 0,5. Siła odejścia od progu przełącza odczyt między dostarczoną predykcją a prywatnym pomiarem. Tylko znaki opuszczają referencję. W mechanizmie bezpośrednim podmiana znaku przekazanego wejścia zmienia wyjście wtedy, gdy operator wykorzystuje to wejście. W mechanizmie wspólnej przyczyny referencja niezależnie odczytuje zachowaną kopię tego samego pomiaru pierwotnego. Podmiana opublikowanego wyniku pozostawia jej wejście bez zmian. W obu modelach normalne wyjścia są identyczne dla tych samych przykładów.

Na każdym rekordzie interwencyjnym algorytm widzi oryginalną odpowiedź referencji, powtórzenie na niezmienionym wejściu oraz powtórzenie po odwróceniu dostarczonych predykcji. Prywatny pomiar i przypadek pozostają te same. Cztery rekordy odpowiadają 24 odczytom dwóch referencji, wliczając oryginały. Osobne 32 nieetykietowane rekordy służą do wyboru drzewa po wskazaniu referencji. Algorytm podejrzewa obserwatora z większą nadwyżką zmian po interwencji nad zmianami po zwykłym powtórzeniu. Przy remisie stosuje równą mieszaninę obu drzew.

## Szum i kontrola pozornej interwencji

| Mechanizm i szum odczytu | Brier po 4 podmianach | Prawidłowe wskazanie | Błędne wskazanie | Wstrzymanie |
|---|---:|---:|---:|---:|
| Bezpośredni, powtarzalny | 0,0953 | 98,46% | 0,00% | 1,54% |
| Wspólny pomiar, powtarzalny | 0,1446 | 0,00% | 0,00% | 100,00% |
| Bezpośredni, 10% niezależnych zmian znaku odczytu | 0,1200 | 76,83% | 9,00% | 14,17% |
| Wspólny pomiar, 10% niezależnych zmian znaku odczytu | 0,1629 | 32,40% | 31,75% | 35,85% |

Ostatni wiersz podaje zgodność wskazanego indeksu z indeksem obserwatora o wspólnej przyczynie. Ponieważ interwencja nie wpływa wtedy na żaden wynik, każde twierdzenie o bezpośredniej zależności jest fałszywym alarmem. W tym sensie częstość fałszywej deklaracji wynosi 64,15%. Porównanie dwóch zwykłych powtórzeń przy tym samym szumie i czterech rekordach dało Brier 0,1645. Odejmowanie szumu powtórzenia nie gwarantuje kontroli błędu. Przy 16 rekordach, szumie 10% i prawdziwej zależności prosty licznik zmian przy podmianie dał 0,1014 wobec 0,1037 po korekcie o powtórzenie. Kryterium dowolnej zmiany osiągnęło 0,1427, ponieważ niemal oba kanały zawsze zmieniały się przynajmniej raz.

W eksperymencie sprawna referencja nie reaguje na podmieniane wejście. W rzeczywistym systemie przydatna referencja również może korzystać z tego wejścia. Wykazana reakcja identyfikuje zależność funkcjonalną. Przypisanie jej szkodliwego wpływu wymaga dalszych założeń lub oceny na etykietach. Możliwość ponownego uruchomienia obserwatora na zachowanym przypadku i podstawienia dokładnie określonej predykcji jest częścią projektu.

## Odtworzenie

Kod wymaga badawczych checkoutów FusionBench i OKAPI obok siebie.

```bash
cd fusionbench
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.intervention_witness --output docs/ensemble-study/intervention-witness/results
PYTHONPATH=.:../okapi python -m pytest -q tests/test_intervention_witness.py
```

Protokół sprzed wykonania testu znajduje się w `protocol.md`. Wyniki obejmują 12 niezależnych banków i 200 partii na reżim w każdym banku. Osobne seedy służą do trenowania drzew, kalibracji i docelowego kontekstu. W tabelach `risks.csv` i `means.csv` pola `correct_witness` i `misidentify` mierzą zgodność indeksu z oznaczonym generatywnie obserwatorem, a `abstain` oznacza remis. Czas obliczeń i czas całej rundy zapisano w dzienniku.
