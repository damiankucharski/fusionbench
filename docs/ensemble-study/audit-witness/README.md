# Jaką informację traci zwycięskie drzewo?

Na 15% rekordów, na których najlepsze walidacyjne drzewo OKAPI było najmniej pewne, jego trafność wynosiła **67,1%**. Liniowy odczyt korzystający z wyniku drzewa oraz siedmiu pierwotnych logitów ekspertów osiągnął **76,6%**. Prerejestrowana poprawa wyniosła **9,54 pp [6,76; 12,32]** w ośmiu nowych niezależnych bankach. W czterech bankach pierwszej serii trafność wzrosła z 67,0% do 74,6%.

| Dostępna informacja dla odczytu logistycznego | Trafność ogółem | Trafność w dolnej warstwie | Brier w dolnej warstwie | Moc audytu 16 etykiet przy zmianie w tej warstwie |
|---|---:|---:|---:|---:|
| Wynik zwycięskiego drzewa | 91,7% | 67,1% | 0,218 | 58,7% |
| Wyniki ośmiu drzew | 92,3% | 70,6% | 0,197 | 71,9% |
| Siedem logitów ekspertów | 93,1% | 75,8% | 0,165 | 91,5% |
| Zwycięzca i siedem logitów | 93,3% | 76,6% | 0,159 | 92,2% |
| Zwycięzca i logity po permutacji | 91,6% | 66,2% | 0,218 | 52,6% |

W porównaniu z decyzją zwycięzcy odczyt obejmujący pełną informację naprawił 19,1% przypadków w dolnej warstwie i dodał błędy w 9,5%. Po przetasowaniu całych wektorów logitów w 40 przedziałach kwantylowych wyniku zwycięzcy zysk zniknął. Przedziały mają skończoną szerokość, więc jest to przybliżona kontrola warunkowa. Rezultat wskazuje, że logity rozróżniają przypadki, którym drzewo przypisuje podobne wyniki. Uczenie dodatkowego liniowego odczytu nad bankiem drzew także odzyskało część tej informacji.

### Audyt z tymi samymi etykietami

Każdy bank obejmował osiem kandydatów z rzeczywistego wyszukiwania GP. Wszystkie odczyty uczono na tych samych 2048 dodatkowych etykietach. Dopiero później wyznaczano test i audyty na osobnych rekordach. Odczyt nie zmieniał wyboru rekordów do oznaczenia. Z każdej partii 256 przykładów wybierano osiem o niskiej, cztery o średniej i cztery o wysokiej pewności zwycięskiego drzewa. Po odwróceniu etykiet w dolnej warstwie te same 16 etykiet wystarczyło do wykrycia 92,2% zmian przez licznik błędów rozszerzonego odczytu wobec 58,7% dla zwycięzcy. Na czystych partiach oba mierzone odsetki fałszywych alarmów wyniosły odpowiednio 5,10% i 4,74%. Przy 15% niezależnych odwróceń moc wyniosła odpowiednio 36,6% i 30,5%.

Progi każdego alarmu ustalono oddzielnie na 399 niezależnych czystych partiach z nominalnym poziomem 5%. Każda ocena testowa korzystała z 512 partii w każdym banku. Oznacza to dodatkowy koszt historycznych czystych etykiet do kalibracji alarmów. Wyjściowe drzewa ograniczono do 10 pokoleń i 384 przykładów do wyszukiwania; dodatkowe 2048 źródłowych etykiet wykorzystały zarówno rekalibracja zwycięzcy, jak i pozostałe odczyty. Wynik nie wyznacza granicy jakości wszystkich drzew OKAPI ani opłacalności modelu na rzeczywistych danych.

Następne pytanie dotyczy konkretnego mechanizmu przewagi logitów. W tym generatorze sygnał jednego eksperta podlega losowemu odwróceniu, a pozostałe mają skorelowane błędy. Znany model generujący pozwala wyprowadzić dokładną optymalną regułę, osobno zbadać wpływ obu mechanizmów i sprawdzić, jak prosty model może nauczyć się tej reguły z etykiet.

### Odtworzenie

[Protokół przed obliczeniami](https://github.com/damiankucharski/fusionbench/blob/f24d621/docs/ensemble-study/audit-witness/protocol.md)

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.audit_witness --out witness-results
```

`predictive.csv` rozdziela wyniki według banku i grupy, `power.csv` podaje liczbę wykrytych partii, a `config.json` zawiera przedział sparowanej różnicy i czas obliczeń 15,49 s. W `timing.json` zapisano ramy czasu pracy obejmującego także protokół, implementację i raport.
