# Ile neuronów potrzeba do odtworzenia puli i jak sprawdzić błąd?

W głównym warunku cztery scalone neurony odtworzyły pulę 43 kandydatów ze średnim RMSE **0,00044** prawdopodobieństwa. Pozostawienie czterech najwyżej ważonych kandydatów dało **0,01605**. Dodatkowa nadwyżka Brier po scaleniu wyniosła −0,0000056 [−0,0000122; +0,0000010] na 16 świeżych bankach. Dla każdego badanego wejścia można obliczyć z góry górną granicę odchylenia od pełnej puli. Przy ośmiu ekspertach mierzących osiem różnych kierunków te same cztery neurony dawały RMSE 0,104. Algorytm wybierający szerokość na podstawie wymaganej granicy zachował wtedy wszystkie osiem.

## Skąd bierze się gwarancja

Dla sigmoidy maksymalna bezwzględna druga pochodna wynosi 1/(6√3). W grupie ekspertów o losowym logicie L (losujemy według wag grupy), rozwinięcie Taylora wokół m=E[L] daje liniowy składnik o średniej zero. Dla pojedynczego wejścia zachodzi

```text
| E[σ(L)] − σ(E[L]) | ≤ Var(L) / (12√3).
```

Przed przybliżaniem dokładnie łączymy wszystkie stałe predykcje w jedną stałą i dodajemy wagi identycznych funkcji. Po podziale pozostałych kandydatów na grupy sumujemy granice z wagami grup. Jeśli θ=(w,b) jest wektorem współczynników, a u=(x,1) wejściem rozszerzonym, wariancja logitu w grupie wynosi uᵀCov(θ)u. Połączenie dwóch grup o masach a,b i środkach θ₁,θ₂ zwiększa łączną macierz granicy o

```text
[a·b / ((a+b)·12√3)] (θ₁−θ₂)(θ₁−θ₂)ᵀ.
```

OKAPI buduje hierarchię Warda, w każdym kroku wybierając połączenie o najmniejszym wzroście oczekiwanej granicy w rozkładzie wejściowym opisanym średnią i kowariancją. Te statystyki wpływają na wybór grup. Nierówność pozostaje punktowa również dla wejść spoza tego rozkładu. Przy wnioskowaniu można uruchomić samą zwartą sieć albo dodatkowo obliczyć granicę z zapisanej kwadratowej formy. To drugie kosztuje dodatkowe operacje, szczególnie w dużym wymiarze. Do budowy nie potrzeba docelowych predykcji nauczyciela ani destylacji.

## Potwierdzenie i kontrola trudności

| Warunek | RMSE 1 neuronu | RMSE 4 scalonych | RMSE 4 największych | Średnia liczba neuronów przy docelowej granicy treningowej 0,01 |
|---|---:|---:|---:|---:|
| Słaby sygnał, 16 etykiet do ważenia puli | 0,01264 | 0,00044 | 0,01605 | 1,52 |
| Silny sygnał, jeden skorelowany duet | 0,00039 | 0,00001 | 0,00168 | 1,00 |
| Brak sygnału, prior 0,75 | 0,00567 | 0,00005 | 0,00106 | 1,27 |
| Dwa odległe progi na jednej osi | 0,28862 | 0 (dwa rzeczywiste składniki) | 0 | 2,00 |
| Osiem różnych kierunków | 0,21223 | 0,10403 | 0,18224 | 8,00 |

W słabym sygnale pojedyncza jednostka miała gwarancję błędu prawdopodobieństwa co najwyżej 0,05 dla 91,53% rekordów; rzeczywisty błąd mieścił się w tym zakresie dla 96,95%. Przy czterech scalonych jednostkach certyfikat objął wszystkie odłożone rekordy. W przypadku ośmiu mocnych kierunków i czterech neuronów tylko **0,59%** rekordów miało gwarancję, choć w tolerancji faktycznie mieściło się **36,47%**. Granica potrafi być bardzo zachowawcza. Największe odchylenie od jej implementacji we wszystkich porównaniach wyniosło 6,7×10⁻¹⁶ w arytmetyce zmiennoprzecinkowej.

Przy kryterium średniej nieskróconej granicy treningowej ≤0,01 słaby sygnał wymagał średnio 1,52 neuronu i dał RMSE 0,00362 na niezależnym teście. W pełnej rodzinie znajdowało się 19 funkcji stałych i 24 stany EM. Część stanów była dokładnie identyczna lub miała wagę reprezentowaną numerycznie jako zero. Po dokładnym pierwszym kroku zostało średnio 24,25 składników w słabym warunku i 15,13 w silnym. Porównanie z k największymi dotyczy tej samej dokładnie zredukowanej reprezentacji.

## Zastosowanie

```python
from okapi.sigmoid_merge import merge_sigmoids

models = merge_sigmoids(W, b, ensemble_weights, feature_mean, feature_covariance,
                        widths=[1, 2, 4, 8])
predictions = models[4].predict(X)
absolute_error_bound = models[4].error_bound(X)
```

Pliki banks.csv i draws.csv.gz zawierają obserwacje i szerokości dla trzech replik etykiet w banku. Podstawowy model zwraca średnią ważoną sigmoid na surowych wejściach; granica wyznacza zgodność z oryginalną pulą, a użyteczność samej puli zależy od zadania. Końcowy przebieg trwał 13,05 s na CPU, 26,09 s łącznie z powtórzeniem po uodpornieniu arytmetyki wag. Następny test rozszerzy pytanie na klasyfikację wieloklasową, z uwzględnieniem tego, że każdy ekspert może dowolnie przesunąć wszystkie swoje logity o tę samą wartość.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m fusionbench.eval.width_certificate --out width-results
```

[Przedrejestracja](https://github.com/damiankucharski/fusionbench/blob/f0b99f8/docs/ensemble-study/width-certificate/protocol.md) · [Metoda OKAPI](https://github.com/damiankucharski/OKAPI/blob/899f4dc/okapi/sigmoid_merge.py)
