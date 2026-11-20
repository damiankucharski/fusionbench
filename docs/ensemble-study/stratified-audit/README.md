# Czy sposób doboru 16 etykiet pomaga wykryć lokalną zmianę?

W ośmiu niezależnych bankach OKAPI przeznaczenie połowy budżetu na najmniej pewne predykcje poprawiło wykrywanie zmian w tej grupie. Po odwróceniu wszystkich etykiet w najmniej pewnych 15% rekordów licznik błędów zaalarmował w **29,7%** partii przy sztywnym podziale 8/4/4 i **12,1%** przy prostym losowaniu 16 rekordów. Prerejestrowana różnica wyniosła **17,6 pp**, z 95% przedziałem t **0,4–34,8 pp** dla sparowanych średnich ośmiu banków. W czterech osobnych bankach pierwszej serii wyniki wynosiły 54,7% i 19,1%. W potwierdzeniu wykrywalność sztywnego podziału w poszczególnych bankach wynosiła od 5,9% do 69,1%.

| Zmiana etykiet | Losowanie proste | Losowanie według niepewności | Losowe liczebności 50/25/25% | Sztywne liczebności 8/4/4 |
|---|---:|---:|---:|---:|
| Brak zmiany (fałszywe alarmy) | 5,2% | 4,7% | 4,1% | 4,2% |
| Wszystkie w dolnej warstwie pewności | 12,1% | 22,2% | 25,3% | 29,7% |
| Połowa w dolnej warstwie | 8,5% | 11,5% | 11,8% | 14,1% |
| Wszystkie w górnej warstwie | 55,0% | 7,4% | 67,7% | 85,0% |
| Wszystkie w środkowym pasie | 51,8% | 14,0% | 10,7% | 15,1% |
| Niezależnie 15% rekordów | 44,4% | 19,1% | 19,2% | 25,7% |

Tabela pokazuje częstość alarmu licznika błędów przy 16 etykietach i wyjściowym poziomie 5%. Pomiędzy metodami zmieniono wyłącznie wybór rekordów oraz próg alarmowy kalibrowany na czystych partiach z tym samym sposobem pobierania. Wagi losowania według niepewności wynosiły `0.01 + p(1−p)`. W próbie tej znalazło się średnio 46,0% przykładów z dolnej warstwy i 2,1% z górnej. Losowy i sztywny podział miały prawie identyczny średni udział dolnej warstwy, 50,0%, lecz wariancja liczby takich rekordów w partii wynosiła odpowiednio 3,92 i 0. Zwiększenie budżetu do 32 podniosło wykrywalność zmiany w dolnej warstwie do 43,3% dla sztywnych limitów i 16,8% dla losowania prostego.

W tym samym warunku przy 16 etykietach iloraz wiarygodności ze stałą alternatywą 15% niezależnych odwróceń wykrywał tylko 7,2% partii z limitami i 4,6% z losowaniem prostym. Przy niezależnym odwróceniu 15% etykiet lepsze było proste losowanie z ilorazem wiarygodności (74,7%) niż sztywne limity z tym samym testem (68,7%). Wyniki określają moc alarmu, a częstość błędów w nadpróbkowanej warstwie wymaga osobnego ważenia dla oszacowania błędu całego wdrożenia.

## Projekt i odtworzenie

Każda partia zawiera 256 nieetykietowanych kandydatów i udostępnia 8, 16 lub 32 ich etykiety. Użyto 12 banków po osiem rzeczywistych wyszukiwań GP. Drzewa, ustawienia i wszystkie progi określono przed oceną na 512 niezależnych partiach testowych na bank. Dla każdej metody i budżetu 399 odrębnych czystych partii posłużyło do kalibracji z losowym rozstrzyganiem remisów. Empiryczne odsetki fałszywych alarmów przez wszystkie 24 kombinacje metod, statystyk i budżetów w potwierdzeniu rozciągały się od 3,9% do 5,5%. Koszt czystych etykiet do kalibracji jednego projektu z budżetem 16 wynosił 6384 na bank. Na danych z rzeczywistych wdrożeń założenia o stabilnym rozkładzie referencyjnym wymagają osobnej kontroli.

Eksperyment wykazał dużą różnicę między bankami w wykrywalności zmian blisko progu decyzji. Kolejny test sprawdzi, czy w tej samej podgrupie pozostałe drzewa lub predykcje wejściowych ekspertów zachowują informację o etykiecie utraconą przez zwycięskie drzewo.

Obliczenia samego eksperymentu trwały 11,77 s, w tym 4,45 s pierwszej serii i 7,29 s niezależnego potwierdzenia. Granice czasu całej pracy i chwilę gotowości raportu zapisuje `timing.json`. [Protokół opublikowany przed wykonaniem](https://github.com/damiankucharski/fusionbench/blob/565a93d/docs/ensemble-study/stratified-audit/protocol.md)

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.stratified_audit --out stratified-results
```

`means.csv` podaje moc dla każdego scenariusza i budżetu. `power.csv` zawiera liczbę alarmów i partii dla każdego banku, `sampling.csv` udziały warstw w próbach, a `models.csv` opis drzew i progów.
