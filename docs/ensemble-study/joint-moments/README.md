# Jak z nieetykietowanych pomiarów odzyskać siedem parametrów jednej sieci?

Sześć czujników o znanym dodatnim kierunku sygnału wystarczyło do odtworzenia liniowej reguły decyzyjnej mimo nieznanych wzmocnień, przesunięć, proporcji klas i jednej pary wspólnie zakłóconych urządzeń. W 16 nowych bankach, przy 1024 obserwacjach, model z median momentów osiągnął **96,62% trafności** i nadwyżkę oczekiwanego Brier **0,00180**. Na tym samym budżecie prawidłowo wyspecyfikowana klasyczna mieszanina Gaussa była jeszcze lepsza, z 96,75% i 0,00077. Wszystkie uczone reguły wdrażamy jako jeden wyraz wolny i sześć współczynników jednej sigmoidy działającej na surowych pomiarach.

## Co identyfikują wspólne obserwacje

Niech π=P(Y=1), Δᵢ=E[Xᵢ|Y=1]−E[Xᵢ|Y=0]>0, a v=π(1−π). Dla trzech warunkowo niezależnych czujników kowariancje Cᵢⱼ=vΔᵢΔⱼ. Wspólny trzeci moment centralny Kᵢⱼₖ=v(1−2π)ΔᵢΔⱼΔₖ. Stąd

```text
r = Kᵢⱼₖ / sqrt(Cᵢⱼ Cᵢₖ Cⱼₖ)
π = (1 − r / sqrt(4 + r²))/2
Δᵢ = sqrt(Cᵢⱼ Cᵢₖ / (Cⱼₖ π(1−π)))
```

Iloraz r pozostaje ten sam po dowolnych przesunięciach i dodatnich przeskalowaniach poszczególnych kanałów. Wśród 20 trójek sześciu kanałów tylko cztery zawierają oba końce jednej zależnej pary. Wśród dziesięciu trójek używanych do oszacowania konkretnego Δᵢ takich trójek jest najwyżej cztery. Przy dodatnich sygnałach i dokładnych momentach większość rekonstruuje tę samą prawidłową wartość, więc mediana jest odporna na nieznaną lokalizację tej pary. W skończonej próbie potrzebne są mierzalne korelacje i stabilna estymacja trzeciego momentu.

Z prawa całkowitej kowariancji odzyskujemy resztową macierz Σ=Cov(X)−vΔΔᵀ. Dla dwóch klas Gaussa o tej samej macierzy reguła Bayesa jest pojedynczą sigmoidą o wagach Σ⁻¹Δ. Wyraz wolny można odzyskać ze średniej X i π. Test jednostkowy sprawdza dokładność tej rekonstrukcji na momentach populacyjnych przy tolerancji 10⁻¹² w głównym układzie. To szczególny przypadek estymacji klas ukrytych z momentów. Założenia o rodzinie błędów i dodatnim kierunku każdego kanału są istotną informacją zewnętrzną.

## Jedna zależna para, pełny wspólny szum i małe budżety

| Warunek, 1024 obserwacje | Źródło zamrożone | Pierwsza trójka | Mediana, diagonalna Σ | Mediana, pełna Σ | Dwa gaussy, pełna wspólna Σ |
|---|---:|---:|---:|---:|---:|
| IID | 0 | 0,00127 | 0,00039 | 0,00050 | 0,00050 |
| Zmiana proporcji klas, skal i zer | 0,00305 | 0,00137 | 0,00050 | 0,00058 | 0,00046 |
| Dodatkowo jeden zależny duet | 0,00646 | 0,09630 | 0,00846 | 0,00180 | 0,00077 |
| Wspólny szum wszystkich | 0,03438 | 0,08327 | 0,10030 | 0,10068 | 0,00997 |
| Cztery dziesiąte siły sygnału | 0,02356 | 0,03854 | 0,01341 | 0,01418 | 0,04093 |
| Identyczne pomiary duetu, odwrócone 25% etykiet | 0,06233 | 0,11207 | 0,06324 | 0,05855 | 0,05694 |

Wartości w tabeli to nadwyżki oczekiwanego Brier nad znanym optymalnym posteriorem. Przy jednej zależnej parze mediana zmniejszyła stratę wobec pierwszej zdatnej trójki o **0,09449 [0,09087; 0,09812]**. Pierwsza trójka w tym eksperymencie celowo zawierała błędną parę. Zachowanie pełnej resztowej kowariancji dało dodatkową korzyść. Mieszanina Gaussa ze wspólną kowariancją poprawiła względem mediany Brier o **0,00103 [0,00034; 0,00172]**. Przy 128 przykładach i jednym skorelowanym duecie nawet ona miała stratę 0,00942, gorszą od zamrożonego źródła 0,00646; mediana miała 0,01978.

Gdy szum wszystkich czujników jest skorelowany, nie ma czystej większości trójek i estymator medianowy zawodzi. Mieszanina Gaussa pozostaje prawidłowym modelem tego przypadku, lecz może źle dopasować słabo rozdzielone klasy. Po zmniejszeniu separacji do 40% mediana przy 1024 przykładach osiągnęła 0,01418 wobec 0,04093 mieszaniny dopasowanej z trzech startów k-means. Przy 128 przykładach obie metody były dużo gorsze od zamrożenia. Następny eksperyment powinien oddzielić błąd optymalizacji mieszaniny od braku informacji o klasach, np. porównać inicjalizację z momentów i log-wiarygodność na oddzielnych nieetykietowanych pomiarach z właściwym błędem klasyfikacji.

Dodatkowa kontrola zmienia jedynie P(Y|X) ze starego q na q′=0,25+0,5q. Wszystkie nieetykietowane wejścia są bitowo identyczne i ten sam model musi zwrócić tę samą predykcję. Dla dowolnego predyktora co najmniej jeden z dwóch warunków ma nadwyżkę Brier przynajmniej E[(q−q′)²]/4. Na wejściach szesnastu banków granica wynosi 0,01416. W tej parze światów dokładna klasyfikacja obu wymaga informacji o etykietach lub założenia wykluczającego jedną z możliwości.

## Odtworzenie

Obliczenia trwały 2,94 s na CPU. Użyto czterech banków rozwojowych, szesnastu banków potwierdzających i po 8192 nowych pomiarów do oceny każdego warunku. Plik linear_models.json zawiera kompletne parametry wszystkich medianowych reguł z budżetem 1024. Każdą można wykonać bez ensemble’u i bez modeli źródłowych jako `expit(X @ weights + intercept)`. Nie używaliśmy destylacji ani treningu GP. Niezależny generator i empiryczne założenia na rzeczywistych zespołach pozostają do sprawdzenia.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m fusionbench.eval.joint_moments --out joint-results
```

[Protokół sprzed obliczeń](https://github.com/damiankucharski/fusionbench/blob/03c0265/docs/ensemble-study/joint-moments/protocol.md)
