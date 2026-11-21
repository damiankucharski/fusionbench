# Ile końcowych etykiet pozwala odróżnić dwie możliwe reguły zespołu?

Dwie możliwe reguły mogą dawać różne etykiety na **12,5% przykładów testowych**, lecz na zaledwie **0,03754% przykładów treningowych**. W rozpatrywanym modelu 512 losowych etykiet pozostawia oba rozwiązania zgodne z treningiem z prawdopodobieństwem 82,51%. Nawet uczeń znający obie możliwe reguły potrzebuje 7978 losowych etykiet, aby mieć 95% szans na zaobserwowanie rozstrzygającego przykładu. Próg dotyczy konkretnej zmiany rozkładu wejść.

### Wzór

Dwa równie prawdopodobne światy mają takie same surowe wejścia i dwa wspólne głosy. Trzeci głos ma kierunek v albo v cos θ + w sin θ. Kierunki są jednostkowe, v i w prostopadłe do siebie i pozostałych dwóch głosów. Podczas treningu współrzędna wzdłuż v ma losowy znak i wielkość chi z k stopniami swobody. Pozostałe są niezależnymi standardowymi gaussami. W teście wszystkie są gaussowskie.

Reguły większości mogą się różnić tylko wtedy, gdy wspólne głosy są przeciwne, co zdarza się z prawdopodobieństwem 1/2. Dla rozkładu treningowego różnica alternatywnych głosów zachodzi, gdy iloraz standardowej normalnej i niezależnej chi(k)/√k jest mniejszy niż −√k cot θ. Na izotropowym teście o rozbieżności decyduje kąt między osiami. Dla dystrybuanty F rozkładu t z k stopniami swobody otrzymujemy

```math
p_{train}=\tfrac12 F_{t_k}(-\sqrt{k}\cot\theta),\qquad p_{test}=\theta/(2\pi).
```

Po n bezszumowych etykietach prawdopodobieństwo nierozstrzygnięcia wynosi r = (1−p_train)ⁿ. Pierwsza rozróżniająca etykieta identyfikuje świat, a przy jej braku oba światy są nadal jednakowo prawdopodobne. Minimalne ryzyka przy tych założeniach wynoszą odpowiednio r/2 dla mylnego wyboru świata, rp_test/2 dla błędnej klasy w teście i rp_test/4 dla binarnego Brier. Ostatnia wartość wykorzystuje predykcję 0,5 tam, gdzie oba nadal możliwe światy się nie zgadzają. Same nieetykietowane wejścia mają identyczny rozkład w obu światach.

| θ = 45°, 512 etykiet | Rozbieżność w treningu | Brak rozstrzygnięcia | Minimalny Brier na teście | Etykiety do 95% szans rozstrzygnięcia |
|---|---:|---:|---:|---:|
| k = 1 | 12,5% | < 10⁻²⁷% | < 10⁻³⁰ | 23 |
| k = 7 | 0,8286% | 1,41% | 0,00044 | 361 |
| k = 15 | 0,03754% | 82,51% | 0,02578 | 7978 |
| k = 31 | 0,0001054% | 99,95% | 0,03123 | 2 841 785 |

### Jedna sieć może zachować niepewność

Niech a,b oznaczają dwa wspólne bity, a c₀,c₁ możliwe trzecie głosy. Obliczamy q = (1−w)c₀ + wc₁ dla posterioru w drugiego świata. Funkcja clip(a+b+q−1, 0, 1) daje prawdopodobieństwo Bayesa. Wystarczą cztery neurony progowe i dwa ReLU realizujące obcięcie. Obie potencjalne osie są tu podane. Taki model unika powtórnego liczenia wspólnych głosów i daje dokładnie te same prawdopodobieństwa co uśrednienie dwóch światów, bez uczenia na prognozach nauczyciela. Nie gwarantuje wyboru jednej prawdziwej osi w nierozstrzygniętych przypadkach.

### Niezależny sprawdzian numeryczny

W głównej komórce (θ=45°, k=15, n=512) pierwsza symulacja 1536 oddzielnych prób treningowych dała binarny Brier 0,02525 wobec dokładnego 0,02578 oraz 80,60% nierozstrzygniętych prób wobec 82,51%. Przedział Monte Carlo dla odsetka nierozstrzygnięcia wyniósł 78,98–82,21%. Zarejestrowaliśmy osobno większe powtórzenie, aby sprawdzić tę różnicę.

W 24 576 nowych próbach Brier wyniósł 0,02598, a Brier sztywnego wyboru jednego świata 0,05240. Nierozstrzygniętych było 83,07% (95% przedział Monte Carlo 82,60–83,54%). Pierwsza i druga estymacja tego odsetka minęły dokładną wartość w przeciwnych kierunkach. W większej próbie teoretyczna wartość Brier leżała też nieznacznie pod dolną granicą przedziału Monte Carlo (0,02581–0,02615). W obu symulacjach różnica obliczeń między pojedynczą siecią a średnią posteriorową wyniosła dokładnie zero. Dalsze wnioskowanie o częstościach w tym prostym modelu opieramy na wzorze.

```sh
python -m fusionbench.eval.majority_coverage --output /tmp/majority-coverage
python -m fusionbench.eval.majority_coverage --confirm --output /tmp/majority-coverage-confirmation
```

[Protokół pierwotny](protocol.md) · [Protokół dodatkowego powtórzenia](confirmation-protocol.md) · [Wyniki analityczne](results/analytic.csv) · [Symulacja](results/means.csv) · [Nowa symulacja](confirmation/means.csv) · [Czas](results/timing.json)
