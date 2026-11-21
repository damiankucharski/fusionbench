# Wieloklasowa granica całkowitej wariacji

Niech zᵢ(x)=Wᵢx+bᵢ będą logitami K ekspertów na tych samych C klasach. Przy nieujemnych wagach aᵢ o sumie 1 porównujemy P=Σᵢaᵢ softmax(zᵢ) i Q=softmax(z̄), gdzie z̄=Σᵢaᵢzᵢ. Zapiszmy hᵢ=zᵢ−z̄ oraz osc(h)=maxⱼ hⱼ−minⱼ hⱼ. Dla dowolnego ustalonego wejścia zachodzi

```text
TV(P,Q) = ½ Σ_c |P_c−Q_c|
        ≤ min(1, Σᵢ aᵢ osc(hᵢ)² / (12√3)).
```

Stała jest niezależna od C i ostra także dla podzbioru zawierającego wiele klas. Poniżej dowód i prostsza do wykonania postać granicy.

## Krzywizna sumy prawdopodobieństw

Dla dowolnego zbioru klas A rozważmy f_A(z)=Σ_{c∈A} softmax_c(z). Jeżeli A jest pusty lub obejmuje wszystkie klasy, pochodne funkcji znikają. W pozostałych przypadkach ustawmy t=f_A(z) i potraktujmy h_c jako zmienną losową pod rozkładem softmax(z). Niech a i b będą jej średnimi warunkowymi na A i na dopełnieniu, v_A i v_B zaś odpowiednimi wariancjami. Dwukrotne zróżniczkowanie f_A(z+sh) po s w zerze daje

```text
D² f_A(z)[h,h] = t(1−t)[ (1−2t)(a−b)² + v_A − v_B ].
```

Przesunięcie wszystkich h_c o tę samą stałą pozostawia wynik bez zmian. Jeśli wszystkie są równe, pochodna znika. W przeciwnym razie z jednorodności kwadratowej wystarczy założyć h_c∈[0,1]. Wystarczy także oszacować dodatni znak, ponieważ dopełnienie A zmienia znak drugiej pochodnej.

Przy t≥1/2 wyraz z (a−b)² jest niedodatni, stąd pochodna nie przewyższa t(1−t)v_A≤1/16. Przy t≤1/2 połóżmy m=max(a,1−a)∈[1/2,1]. Wtedy (a−b)²≤m² i v_A≤a(1−a)=m(1−m). Wyrażenie w nawiasie nie przewyższa m−2tm². Dla 0<t≤1/4 maksimum po m wypada w m=1. Pozostaje t(1−t)(1−2t), które osiąga 1/(6√3) przy t=(3−√3)/6. Dla 1/4≤t≤1/2 maksimum wypada w m=1/(4t), dając (1−t)/8≤3/32<1/(6√3). Przy h=1 na A i h=0 na dopełnieniu oraz wskazanym t równość jest osiągnięta. Po przywróceniu jednostek daje to

```text
|D² f_A(z)[h,h]| ≤ osc(h)²/(6√3).
```

## Przejście od krzywizny do TV

Rozwinięcie f_A(z̄+hᵢ) wzdłuż odcinka od z̄ ma resztę ograniczoną przez osc(hᵢ)²/(12√3). Po pomnożeniu przez aᵢ i zsumowaniu wszystkie wyrazy liniowe znoszą się, gdyż Σᵢaᵢhᵢ=0. Otrzymujemy jedną granicę na |P(A)−Q(A)| dla każdego A. Ponieważ TV(P,Q)=max_A |P(A)−Q(A)| dla skończonego zbioru klas, obowiązuje ta sama granica. Symetryczna mieszanina dwóch małych przeciwnych zaburzeń wokół punktu o maksymalnej krzywiźnie osiąga tę stałą w granicy. Test weryfikuje stosunek rzeczywistego TV i oszacowania dla 2, 6 i 34 klas.

## Kwadratowa funkcja surowego wejścia

Odejmijmy w każdym ekspercie średni po klasach wektor parametrów θᵢc=(wᵢc,bᵢc). Operacja usuwa dowolne przesunięcie wszystkich logitów jednego eksperta o wspólną funkcję afiniczną. Oznaczmy tak wycentrowane parametry przez θ′ᵢc oraz Dᵢc=θ′ᵢc−Σⱼaⱼθ′ⱼc. Dla u=(x,1) współrzędne (Dᵢu) odpowiadają hᵢ z usuniętą średnią po klasach. Nierówność Cauchy’ego–Schwarza daje osc(v)²≤2‖v−mean(v)‖². Możemy więc raz przygotować macierz

```text
M = [1/(6√3)] Σᵢ aᵢ DᵢᵀDᵢ
TV(P,Q) ≤ min(1, uᵀ M u).
```

Wymaga ona miejsca O((d+1)²) przy d cechach. Wysoki wymiar może sprawić, że sprawdzanie zgodności kosztuje więcej od bezpośredniego uruchomienia małej puli. Sama skompresowana predykcja nie potrzebuje tej macierzy. Wariant z oscylacjami na ogół daje ciaśniejszą granicę, ale oblicza wszystkie odpowiednie logity.

## Co dokładnie wynika dla decyzji i Brier

Niech δ będzie dowolną z powyższych granic skróconą do 1. Każda współrzędna prawdopodobieństwa różni się najwyżej o δ. Jeżeli różnica dwóch najwyższych prawdopodobieństw Q przekracza 2δ, prognozowana klasa jest zgodna z P. Ponadto ‖P−Q‖₂≤√2δ, gdyż dodatnie różnice sumują się do TV, tak samo jak bezwzględne ujemne.

Dla sumarycznego wieloklasowego Brier i P będącego prawdziwym posteriorem nadwyżka Q wynosi ‖P−Q‖₂²≤2δ². Dla dowolnej etykiety y odległości obu rozkładów od e_y są co najwyżej √2, a różnica tych odległości nie przekracza √2δ. Wobec tego moduł różnicy kwadratów odległości jest co najwyżej 4δ−2δ². To samo wynika po uśrednieniu względem dowolnego prawdziwego rozkładu etykiety. Druga granica jest większa, bo obejmuje również przypadek mylącej się puli. Obie granice Brier są osiągalne w prostych dwuklasowych rozkładach.

## Przykład bez dokładnej redukcji do jednego softmaksu

Niech z₁=(x,0,0), z₂=(−x,0,0), a obaj eksperci mają wagę 1/2. Ich pula spełnia P₂=P₃ i P₁ jest parzystą, niestałą funkcją. W zerze P₁=1/3, podczas gdy dla |x|→∞ dąży do 1/2. W pojedynczym afinicznym softmaksie log(P₁/P₂) musiałby być funkcją afiniczną x. Tutaj jest parzysty i niestały. Tego ensemble’u nie da się zatem dokładnie zapisać jednym afinicznym softmaksem na całej osi.
