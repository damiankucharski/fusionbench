# Kiedy 43 sigmoidy można zastąpić jedną bez destylacji?

W słabym sygnale, po ustaleniu wag kandydatów z 16 etykiet, siedmioparametrowy model zbudowany z ich współczynników osiągnął nadwyżkę Brier **0,01313** wobec **0,01255** pełnej puli 43 modeli. Różnica **+0,00058 [−0,00002; +0,00118]** zmieściła się w przedrejestrowanym marginesie +0,002. Na pojedynczym CPU wykonanie dla partii 1024 pomiarów trwało 0,0104 ms, wobec 0,1859 ms przy puli z dokładnie połączonymi stałymi. Kompresja nie uczyła się na predykcjach nauczyciela. Wynik dotyczy określonej rodziny podobnych liniowych klasyfikatorów i średniej po 16 niezależnych bankach.

## Od parametrów puli do pojedynczych wag

Niech f(x)=Σaᵢσ(wᵢᵀx+bᵢ), gdzie wagi aᵢ sumują się do jedności. Wybieramy środek μ równy średniej wejść treningowych i obliczamy pᵢ=σ(wᵢᵀμ+bᵢ), f₀=Σaᵢpᵢ oraz g=Σaᵢpᵢ(1−pᵢ)wᵢ. Parametry

```text
w = g / (f₀(1−f₀))
b = logit(f₀) − wᵀμ
```

sprawiają, że nowa sigmoida ma w tym punkcie dokładnie tę samą wartość i gradient co pula. OKAPI liczy współczynniki w przestrzeni logarytmicznej także przy dużych zgodnych logitach. Testy kontrolują m.in. dokładne odtwarzanie identycznych sigmoid przy biasach ±1000. Wariant sieczny zachowuje kierunek wag, lecz ustawia ich skalę z logitów puli w dwóch syntetycznych punktach μ±Σg/√(gᵀΣg). Do wyprowadzenia pierwszego wariantu wystarczają parametry kandydatów i średnia wejść. Drugi wykorzystuje także macierz kowariancji. Wdrożenie obu wymaga wyłącznie siedmiu liczb i sześciu surowych pomiarów.

## Porównanie z prostszymi odpowiedziami

| Metoda | Nadwyżka Brier, słaby sygnał i 16 etykiet | RMSE predykcji do pełnej puli |
|---|---:|---:|
| 43 składniki, arytmetyczna średnia prawdopodobieństw | 0,01255 | 0 |
| Średnia samych wag i biasów | 0,01266 | 0,01067 |
| Jedna sigmoida z lokalnie zgodnym gradientem | 0,01313 | 0,01028 |
| Jedna sigmoida z siecznej | 0,01323 | 0,01170 |
| Faktyczna destylacja na 1024 treningowych wyjściach puli | 0,01266 | 0,00697 |
| Pojedynczy kandydat z największą wagą | 0,01652 | 0,04207 |
| Regresja logistyczna uczona wyłącznie na 16 etykietach | 0,04233 | 0,19667 |

Przybliżenie lokalne spełniło zaplanowany margines, ale w tej próbie nawet średnia współczynników dawała mniejszą średnią stratę zadaniową. Destylacja najwierniej odtworzyła prawdopodobieństwa puli. Jedna z tych miar może się poprawić, choć druga się pogarsza. Przy 64 etykietach lokalne przybliżenie i pula miały odpowiednio 0,00844 i 0,00850. Przy silnym sygnale i jednej skorelowanej parze wszystkie cztery warianty kompresji osiągały około 0,00054. Bez sygnału i z 16 etykietami otrzymaliśmy 0,01287 dla stycznej, 0,01216 dla puli i RMSE stycznej 0,00457.

## Przykład, który uniemożliwia dokładną redukcję

Uśredniliśmy dwie równo ważone sigmoidy σ(6(x−1)) i σ(6(x+1)). Ich pełna pula z założenia jest prawdziwym posteriorem na standardowym rozkładzie normalnym. Między progami pojawia się płaskie przejście. Symetria wymusza wyraz wolny zero dla dokładnego modelu jednosigmoidalnego, a wykładnicze tempo obu ogonów wymusza nachylenie sześć. Taka sigmoida ma pochodną 1,5 w zerze, podczas gdy pochodna puli wynosi około 0,0148. Dokładny pojedynczy odpowiednik nie istnieje. Średnia wag osiągnęła nadwyżkę Brier 0,08323, styczna 0,05747, sieczna 0,00808, a destylacja przez entropię krzyżową 0,00859. Różnica między dwiema ostatnimi metodami nie jest dowodem pokonania optymalnej regresji Brier, ponieważ minimalizowano różne cele.

## Koszt i odtworzenie

Bank zawierał 24 różne stany EM i 19 stałych kandydatów. W pomiarach czasu zoptymalizowana pełna pula używała jednej stałej i 24 zmiennych składników. Mediana z 17 powtórzeń wyniosła 0,186 ms na partię wobec 0,336 ms w naiwnym kodzie wszystkich 43 oraz 0,0104 ms po redukcji. To pojedynczy lokalny benchmark NumPy, bez przedziału ufności sprzętowego. W głównym warunku średnie czasy samego tworzenia modelu z już gotowych kandydatów wyniosły 0,004 ms dla średniej parametrów, 0,576 ms dla stycznej, 0,889 ms dla siecznej i 1,733 ms dla destylacji. Koszt dopasowania wejściowych kandydatów jest wspólny i pozostał osobno w czasie całego doświadczenia.

Końcowe doświadczenie na CPU trwało 14,31 s. Początkowy optymalizator liniowego komparatora zakończył próbną serię błędem numerycznym. Pełny raport wykorzystuje tłumioną metodę Newtona dla obu wypukłych regresji i nie zawiera częściowych wierszy. Sprawdziliśmy numerycznie eksport parametrów do pierwotnej skali wejść dla wszystkich opublikowanych modeli. Następna runda zweryfikuje na rzeczywistych danych, czy pula logistycznych ekspertów najpierw w ogóle przewyższa silny pojedynczy model, a następnie ile tej poprawy pozostaje po kompresji.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m fusionbench.eval.sigmoid_pool --out sigmoid-results
```

[Przedrejestracja](https://github.com/damiankucharski/fusionbench/blob/948a879/docs/ensemble-study/sigmoid-pool/protocol.md) · [Funkcja OKAPI](https://github.com/damiankucharski/OKAPI/blob/f580e93/okapi/sigmoid_pool.py)
