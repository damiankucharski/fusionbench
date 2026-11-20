# Losowy audyt etykiet przy zmianie niewidocznej w predykcjach

Przy niezależnej zmianie 15% etykiet audyt **16 losowych etykiet z wdrożenia** zaalarmował w 74,0% partii przy użyciu ilorazu wiarygodności i w 61,4% przy użyciu Brier. Wcześniej zarejestrowana różnica wynosi 12,6 pp z 95% przedziałem t 9,0–16,2 pp, liczonym z ośmiu sparowanych średnich dla niezależnych banków. Zwykła strata logarytmiczna osiągnęła 72,8%, liczba pomyłek 44,5%, a kontrola oparta na samych predykcjach 5,3%. Zwiększenie budżetu do 32 etykiet podniosło wykrywalność ilorazu wiarygodności do 90,3%.

## Kalibracja i koszty

Utworzono 8 niezależnych banków po 12 rzeczywistych wyszukiwań OKAPI. Każde wyszukiwanie trwało 10 pokoleń. Dla każdego banku zamrożono zwycięzcę na osobnej walidacji i skalibrowano jego wynik logistycznie na 2048 czystych rekordach. Szacujemy moc jednorazowego audytu pojedynczej partii, przy stabilnym 10-procentowym udziale znanego kontekstu awarii pierwszego eksperta.

Każda statystyka i każdy budżet zostały skalibrowane na oddzielonych od testu 399 czystych partiach. Próg ma ten sam marginalny poziom 5% dla wszystkich metod przy wymiennych czystych partiach. Losowa ranga rozstrzyga remisy również dla dyskretnej liczby pomyłek. Następnie oceniono empiryczne fałszywe alarmy na 512 dodatkowych czystych partiach w każdym banku. Wyniosły one od 4,2% do 5,6% w całej siatce metod i budżetów. Przy 16 etykietach wartości dla ilorazu wiarygodności i Brier wynosiły 5,10% i 4,66%.

Koszt 16 etykiet dotyczy jednej monitorowanej partii. Kalibracja progu dla tego budżetu wykorzystała 6384 czyste historyczne etykiety na bank. Progi dla wszystkich zagnieżdżonych budżetów łącznie wymagały 25 536 unikalnych czystych etykiet na bank. Symulator pozwalał wygenerować je tanio. Skuteczność przy skromnej historycznej kalibracji pozostaje osobnym pytaniem.

## Moc audytu ilorazem wiarygodności

| Zmiana etykiet | 4 etykiety | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| Czyste dane (fałszywe alarmy) | 5,1% | 4,8% | 5,1% | 4,8% | 4,2% |
| Niezależnie 5% | 16,7% | 24,4% | 36,5% | 50,2% | 68,3% |
| Niezależnie 15% | 37,8% | 54,0% | 74,0% | 90,3% | 98,9% |
| Niezależnie 40% | 74,4% | 90,4% | 98,6% | 100,0% | 100,0% |
| Wszystkie etykiety | 99,7% | 100,0% | 100,0% | 100,0% | 100,0% |
| Górne 15% pewności | 50,8% | 73,9% | 93,3% | 99,4% | 100,0% |
| Dolne 15% pewności | 5,4% | 5,2% | 6,0% | 6,0% | 6,2% |

Statystyka sumuje logarytm stosunku prawdopodobieństwa zaobserwowanej etykiety w modelu z symetryczną zmianą 15% etykiet do prawdopodobieństwa w modelu czystym. Wartość 15% ustalono przed uruchomieniem. Każda metoda otrzymywała te same predykcje i te same losowe etykiety.

W warunku najniższej pewności zmieniło się 15,3% etykiet. Błąd klasyfikacji wzrósł z 8,5% do 13,9%, a Brier z 0,0651 do 0,0794. Przy 64 etykietach liczba pomyłek wykryła 38,8% takich partii, Brier 14,7%, a iloraz wiarygodności 6,2%. Kiedy prawdopodobieństwo przypisane klasie zbliża się do 0,5, odwrócenie etykiety słabo zmienia jej stratę logarytmiczną. Wartość audytu zależała więc zarówno od liczby zmienionych etykiet, jak i od tego, gdzie wystąpiła zmiana. Dla kontroli bez etykiet sprawdzono identyczność decyzji czysta–zmieniona w każdej parze i przy każdym budżecie.

To doświadczenie mierzy alarm na jednym syntetycznym mechanizmie bazowym. Nie mierzy jakości drzew po interwencji ani łącznej częstości choć jednego fałszywego alarmu po wielokrotnym monitorowaniu. Następny sprawdzian powinien utrzymać całkowity budżet i poziom alarmów oraz porównać losowe pobieranie etykiet ze stratyfikacją według pewności, ze szczególnym uwzględnieniem zmian na granicy decyzji.

## Odtworzenie

[Protokół sprzed obliczeń](https://github.com/damiankucharski/fusionbench/blob/f539e508badcc48e4caa27e4130656b1e71f9061/docs/ensemble-study/sentinel-audit/protocol.md)

Z katalogu zawierającego oba repozytoria z odpowiednimi poprawkami można uruchomić

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.sentinel_audit --out sentinel-results
```

`config.json` zapisuje prerejestrowaną różnicę i parametry. `models.csv` opisuje wybrane drzewa i rekalibrację, `power.csv` liczbę alarmów i partii w każdej komórce każdego banku, `means.csv` średnie z banków, a `risks.csv` rzeczywiste częstości odwróceń i straty.
