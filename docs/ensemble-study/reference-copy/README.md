# Kiedy referencja kopiuje błędy eksperta

W 12 niezależnych próbach wyuczyliśmy po dwa drzewa OKAPI. W każdym z 24 wyszukiwań GP wybrało pojedynczego eksperta odpowiedniego dla źródłowego reżimu. Sprawdziliśmy, czy można wybrać właściwe drzewo na podstawie nieetykietowanych pomiarów w nowej partii. Błąd Briera selektora opartego na zgodności z referencją wzrósł z 0,0948 do 0,2541, kiedy referencja zaczęła poprawnie imitować rozkład *przeciwnego* reżimu. Trafność spadła z 91,97% do 63,01%.

## Dlaczego ten przypadek jest nierozróżnialny

Przy zbalansowanej etykiecie Y każdy czujnik X, o sile r, ma prawdopodobieństwo (1+r)/2 wskazania poprawnego znaku. Dwa główne czujniki mają siły a=0,25 i b=0,85. W drugim reżimie ich role są zamienione. Czysta referencja H ma siłę c=0,75 i szum niezależny od pozostałych czujników warunkowo względem Y. Wygenerowaliśmy zaburzoną referencję, która z prawdopodobieństwem k=0,62118 kopiuje słabszego eksperta. W pozostałych przypadkach mierzy etykietę z siłą 0,17236. Zaburzenie zmienia oba składniki jakości referencji.

Wartości dobraliśmy algebraicznie. Gdy z oznacza (1−k) razy siła samodzielnego pomiaru, zgodności słabszego i mocniejszego eksperta z referencją w skali iloczynów znaków wynoszą odpowiednio az+k i bz+abk. Równania az+k=bc i bz+abk=ac zrównują je ze zgodnościami po *zamianie* jakości ekspertów. Wspólny rozkład (X₁,X₂,H) jest symetryczny względem jednoczesnego odwrócenia wszystkich znaków. Trzy momenty par determinują więc osiem prawdopodobieństw. Dokładną zgodność wszystkich ośmiu komórek sprawdza test jednostkowy. Dowolny selektor korzystający z tych nieetykietowanych trójek widzi ten sam rozkład w dwóch światach, które wymagają przeciwnych decyzji.

## Działanie kontroli

Skalibrowaliśmy ryzyka obu drzew i średnie niezgodności w osobnych grupach, po 384 przykłady z każdego reżimu. Funkcja OKAPI `estimate_context_prevalence` rzutowała obserwowaną niezgodność na odcinek łączący te dwie populacje. Przed wyszukiwaniem zapisaliśmy protokół i budżety. Referencje służyły jedynie do wyboru drzewa. Predykcja przykładu korzystała z dwóch podstawowych pomiarów. Równa mieszanina poniżej uśrednia prawdopodobieństwa obu drzew.

| Wybór przy 32 obserwacjach | Brier, czysta referencja | Brier, pełne zaburzenie | Trafność przy pełnym zaburzeniu |
|---|---:|---:|---:|
| Pierwsza referencja | 0,0948 | 0,2541 | 63,01% |
| Druga, niezależna i sprawna referencja | 0,0943 | 0,0943 | 92,06% |
| Dwaj świadkowie, mieszanina przy niezgodzie | 0,0937 | 0,1453 | 77,54% |
| Druga referencja jest duplikatem pierwszej | 0,0948 | 0,2541 | 63,01% |
| Stała równa mieszanina | 0,1446 | 0,1446 | 77,50% |
| Znany reżim, najlepsze według kalibracji drzewo | 0,0919 | 0,0919 | 92,50% |

Dodatkowa sprawna referencja obniżyła Brier przy pełnym zaburzeniu o 0,15977 z 95-procentowym przedziałem od 0,15891 do 0,16064 po sparowanych próbach. Reguła przejścia do mieszaniny przy niezgodzie dwóch referencji obniżyła Brier o 0,10873 [0,10814, 0,10932]. Sama kopia nie dostarczała informacji. Gdy połowa obserwacji referencji pochodziła z mechanizmu zaburzonego, pierwsza referencja wybierała prawidłowe drzewo w 49,90% partii.

W ograniczonej, dwuekspertowej puli GP odnajdywało tę samą regułę w każdej próbie. Zmienność wyników pochodzi głównie z losowania danych źródłowych i docelowych partii. Mechanizm kopiowania jest konstrukcją kontrolną o dokładnie dobranych parametrach. Wartość dodatkowej referencji zależy od jej jakości oraz pochodzenia jej błędów. Pozostaje pytanie, czy da się ustalić, który z dwóch różnych świadków kopiuje eksperta, gdy oba źródła są traktowane jako potencjalnie zawodne.

## Odtworzenie

Przy checkoutach badawczych obu repozytoriów obok siebie i interpreterze z ich zależnościami:

```bash
cd fusionbench
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.reference_copy --output docs/ensemble-study/reference-copy/results
PYTHONPATH=.:../okapi python -m pytest -q tests/test_reference_copy.py
```

Protokół w `protocol.md` zapisano przed obliczeniami. Wyniki dla seedów, faktyczne drzewa, konfiguracja, średnie i kontrasty znajdują się w `results/`. Każdy z dwóch reżimów otrzymał 200 odrębnych zbiorów kontekstowych na seed. Docelowe ryzyko zapytania zintegrowano dokładnie po możliwych pomiarach i etykiecie. Czasy rundy oraz kontrolę przełącznika odnotowano w dzienniku projektu.
