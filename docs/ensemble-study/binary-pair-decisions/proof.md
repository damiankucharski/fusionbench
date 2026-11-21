# Decyzje dwóch ekspertów i gwarantowane parowania

Niech σ(t)=1/(1+e⁻ᵗ), a klasą dodatnią będzie zdarzenie, że arytmetyczna średnia prawdopodobieństw wynosi przynajmniej 1/2. Dla dwóch ekspertów o równych wagach mamy

```text
σ(a)+σ(b)−1 = σ(a)−σ(−b).
sign(σ(a)+σ(b)−1) = sign(a+b).
```

Druga równość wynika ze ścisłej monotoniczności sigmoidy. Jeśli a i b są afinicznymi funkcjami tych samych surowych cech, jedna sigmoida o uśrednionych współczynnikach i biasie podejmuje zatem te same decyzje na wszystkich wejściach. Tożsamość zachowuje próg klasyfikacji. Na przykład przy logitach 0 i 2 średnia prawdopodobieństw wynosi ok. 0,6904, podczas gdy σ(1) wynosi ok. 0,7311.

## Kiedy rośnie liczba ekspertów

Rozważmy parę o logicie środkowym m=(a+b)/2 i półróżnicy d=(a−b)/2. Wkład pary w odchylenie od połowy wynosi

```text
(σ(m+d)+σ(m−d))/2 − 1/2 = sinh(m) / [2(cosh(m)+cosh(d))].
```

Mianownik jest zawsze dodatni. Wiele par może mieć różne łączne wagi, o ile członkowie każdej pary mają równe udziały. Jeśli na danym wejściu wszystkie m są nieujemne, to średnia wszystkich prawdopodobieństw jest nie mniejsza niż połowa. Jeśli wszystkie m są niedodatnie i chociaż jeden jest ujemny, to średnia jest ściśle mniejsza niż połowa. Przy wszystkich m=0 uzyskujemy dokładnie remis. Jeśli każde m jest dodatnią wielokrotnością tej samej funkcji afinicznej g, jedna granica g(x)=0 globalnie wystarcza do klasyfikacji. Pary o środku stale zerowym również można dodać dowolną liczbę razy przy zachowaniu przynajmniej jednego dodatniego mnożnika. Istotna jest wspólna orientacja granicy.

W implementacji obliczamy trzy średnie logity wprost z sześciu surowych wektorów wag. Ich zgodne znaki wystarczą także wówczas, gdy poszczególne składowe par mają przeciwne znaki. Na niepewnych wierszach obliczamy trzy półróżnice i odzyskujemy sześć logitów jako m±d. W pobliżu numerycznego remisu można dodatkowo posortować skrajne logity i obliczyć znak sumy marginesów w przestrzeni logarytmów. Przykładowo, dla a=999 i b=−1000 typowe obliczenie obu prawdopodobieństw w float64 zwraca dokładnie 1 i 0. Matematyczny znak jest ujemny i można go odzyskać przez porównanie a z −b. Dla dowolnych niemal identycznych liczb zmiennoprzecinkowych pozostaje zwykłe ograniczenie precyzji obliczanych logitów.

## Sortowanie daje pełny test istnienia parowania o zgodnych znakach

Dla posortowanych z₁≤…≤z₂ₘ istnieje pełne parowanie, którego każda suma jest dodatnia, wtedy i tylko wtedy, gdy zᵢ+z₂ₘ₊₁₋ᵢ>0 dla każdego i≤m. Spełnienie warunku daje konkretne parowanie skrajnych wartości. Gdy dla pewnego i warunek zawiedzie, pierwszych i elementów potrzebuje partnerów ściśle za pozycją 2m+1−i, ponieważ żaden element wcześniejszy nie da dodatniej sumy nawet z największym z tych pierwszych i. Takich partnerów jest tylko i−1. Ten sam argument działa dla nierówności nieostrych, a po zmianie znaku także dla sum niedodatnich i ujemnych. Wniosek dotyczy dowodu przez parowanie. Na przykład dla (−10,1,1,1) średnie prawdopodobieństwo wynosi około 0,5483, mimo że żadne pełne parowanie nie ma wszystkich sum nieujemnych.

## Dwie granice stosowalności

1. Przy nierównych wagach 0,55 i 0,45 oraz logitach x i −10x znak średniej względem 0,5 jest znakiem 0,55 tanh(x/2)−0,45 tanh(5x). Funkcja jest nieparzysta. Dla x>0 stosunek tanh(5x)/tanh(x/2) maleje od 10 do 1, osiągając 11/9 raz. Monotoniczność wynika z malejącej w zmiennej c funkcji c/sinh(2cx). Powstają trzy przecięcia z progiem, co uniemożliwia zastąpienie decyzji jedną prostą granicą.
2. Dla trzech równie ważnych logitów 10(x+1), −10x i 10(x−1) średnia kolejno wypada poniżej, powyżej, poniżej i powyżej połowy w punktach −10, −0,5, +0,5 i +10. Są co najmniej trzy przecięcia. Po podstawieniu t=exp(10x) i pomnożeniu przez dodatnie mianowniki otrzymujemy wielomian co najwyżej trzeciego stopnia, więc przecięcia są dokładnie trzy.
