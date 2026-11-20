# Jak odróżnić zmianę współzależności ekspertów od zmiany klas?

Gdy znika wspólny szum sześciu czujników, dwa najsłabsze odzyskują dodatnie wagi. Ujemne wagi pomagały wcześniej skasować szum. Na ośmiu nowych bankach model odzyskał wszystkie sześć znaków po zaniku korelacji. W tym samym teście częstość klasy dodatniej wzrosła z 50% do 80%. Oszacowanie obu zmian z 1024 nieetykietowanych przykładów poprawiło trafność pojedynczego modelu z **91,67% do 98,68%**. Poprawa nadwyżki oczekiwanego Brier wyniosła **0,05258 [0,04929; 0,05588]**.

## Co mierzymy przed ujawnieniem etykiet

Sześć stabilnych pomiarów x ma warunkowe średnie ±s i macierz szumu R=(1−ρ)I+ρ11ᵀ. Dla dowolnej ortonormalnej bazy B prostopadłej do s i jedynek, Bᵀx zawiera tylko szum o wariancji 1−ρ. Średni kwadrat tych czterech kontrastów daje estymator ρ niezależny od udziału klas i wspólnego przesunięcia punktu zerowego wzdłuż s. Siły s estymujemy z 768 etykiet źródłowych. W prostszym estymatorze bez zmiany zera wykorzystujemy E[xxᵀ]=R+ssᵀ. Przy ρ=0 i udziale klasy 80% ten drugi estymator również się sprawdził, z nadwyżką Brier 0,000047 wobec 0,000079 dla kontrastów.

Wagi stabilnych czujników wynoszą 2R⁻¹s. Jeżeli instrumenty nie zmieniły zera, średnia x wzdłuż s pozwala wyznaczyć udział klasy π, ponieważ E[x]=(2π−1)s. Po równoczesnej zmianie ρ i π uzyskaliśmy średnio ρ=0,011, π=0,803. Nadwyżka Brier zamrożonego modelu wynosiła 0,052663. Już z 256 nieetykietowanych przykładów po adaptacji spadła do 0,000127.

| Warunek docelowy, 1024 wejścia | Brier ponad optimum ze źródła | Po adaptacji kontrastów i prioru | Po oszacowaniu zmiany zera z osobnego kanału |
|---|---:|---:|---:|
| Bez zmiany | 0,0002 | 0,0002 | 0,0003 |
| Tylko zanik korelacji | 0,0515 | 0,0001 | 0,0002 |
| Wzrost korelacji do 0,90 | 0,0210 | 0,0002 | 0,0004 |
| Tylko zmiana proporcji klas | 0,0117 | 0,0002 | 0,0007 |
| Zanik korelacji i zmiana klas | 0,0527 | 0,0001 | 0,0004 |
| Przesunięcie bloku stabilnego | 0,0214 | 0,0745 | 0,0003 |
| Przesunięcie wszystkich czujników | 0,0465 | 0,1119 | 0,1134 |

Ostatnie dwa wiersze pokazują problem identyfikacji z pierwszego momentu. Po zmianie zera o 0,6s kontrasty prawidłowo odzyskały ρ=0,652. Średnia bloku wskazała jednak π=0,809 przy rzeczywistych 0,5. Osobny czujnik pozwolił skorygować przesunięcie, dopóki jego własne zero pozostawało stałe. Po przesunięciu wszystkich siedmiu pomiarów oba estymatory udziału klas zgodnie się pomyliły. Te dwa rodzaje zmiany dają zgodne średnie, a pełne rozkłady w symulatorze mogą się jeszcze różnić.

## Co z rankingiem drzew OKAPI?

Osiem prawdziwych startów GP produkuje osiem kandydatów na bank, trenowanych na tych samych etykietach. Oczekiwany Brier kandydata f przy posteriorze q zawiera składnik E[(f−q)²] i wspólną dla kandydatów stałą. Do wyboru seeda wystarczy więc przewidzieć jego odległość od q na nieetykietowanych wejściach. Do selektora nie trafiają etykiety docelowe.

| Kryterium przy jednoczesnej zmianie ρ i π | Spearman z rankingiem testowym | Strata wobec najlepszego seeda |
|---|---:|---:|
| Losowy seed, średnia z ośmiu | — | 0,00770 |
| Brier na 256 źródłowych etykietach walidacyjnych | 0,613 | 0,00325 |
| Zamrożony posterior na tych samych 1024 docelowych wejściach | 0,819 | 0,00201 |
| Oszacowany docelowy posterior | 0,957 | 0,00011 |
| Dokładny posterior na tej samej partii 1024 wejść | 0,959 | 0,00011 |

Dla redukcji straty wyboru wobec walidacji osiem banków daje szeroki przedział 95% od −0,00256 do +0,00884. Wynik rankingowy jest wtórny. Przy przesunięciu wyłącznie bloku stabilnego estymator naiwnie zakładający stałe zero odwraca ranking (Spearman −0,446), podczas gdy wersja z poprawnie wybranym zewnętrznym punktem odniesienia osiąga 0,855. Przy przesunięciu wszystkich kanałów oba warianty wypadają słabo.

## Założenia i odtworzenie

To kontrolowany test o znanej rodzinie rozkładów. Estymator zna numer podatnego na awarię czujnika, jego częstość odwróceń 0,10, jednostkowe wariancje warunkowe i typ wspólnych korelacji. Siły czujników, współzależność i priory dopasowuje z danych. Model działa na siedmiu surowych pomiarach. Drzewa otrzymują odpowiadające im prawdopodobieństwa ekspertów. Żaden z tych wyników nie dowodzi uniwersalnej identyfikowalności zmian rozkładu z predykcji zespołu.

Całość obejmowała cztery banki rozwojowe i osiem oddzielnych potwierdzających, 96 wyszukiwań GP po 10 pokoleń, po 8192 nowych próbek testowych na warunek i wspólne losowania między warunkami. Obliczenia badania na CPU trwały 11,89 s. Pliki models.csv, selectors.csv i candidates.csv zawierają wyniki poszczególnych banków. Następne pytanie dotyczy tego, czy kształt rozkładu pojedynczego czujnika pozwoli rozdzielić dryf zera wszystkich urządzeń od rzeczywistej zmiany proporcji klas.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OKAPI_LOG_LEVEL=ERROR \
python -m fusionbench.eval.covariance_transport --out covariance-results
```

[Protokół zapisany przed doświadczeniem](https://github.com/damiankucharski/fusionbench/blob/d7a67c7/docs/ensemble-study/covariance-transport/protocol.md)
