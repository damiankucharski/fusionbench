# Czy certyfikowane pomijanie ekspertów przyspiesza decyzje?

Odpowiedź zależała od wielkości partii i podobieństwa ekspertów. W jedynym nietrywialnym przypadku przewagi nad prostą kontrolą jednomyślności — pełnej partii HAR z niemal identycznymi modelami — klasyfikacja zajęła **1,502 µs na rekord**, wobec **1,625 µs** z kontrolą jednomyślności i **2,057 µs** przy pełnej puli. Przy pojedynczych rekordach router był wolniejszy od pełnej puli we wszystkich 12 układach. Wszystkie dokładne warianty zwróciły identyczne klasy na każdym sprawdzonym przykładzie i na kontrolach remisów.

## Co wolno pominąć

Pięć ekspertów przewiduje prawdopodobieństwa sześciu lub dziesięciu klas. Pojedynczy model ma logity z̄ będące średnią logitów ekspertów. Odejmijmy od każdego eksperta z̄ i średnią resztę po klasach, otrzymując wektory hᵢ. Energia S=Σᵢ aᵢ‖hᵢ‖² kontroluje dwie wystarczające przesłanki.

- Jeśli różnica pomiędzy pierwszym i drugim prawdopodobieństwem pojedynczego softmaksu przekracza S/(3√3), wcześniej udowodniona granica całkowitej wariacji gwarantuje wspólną zwycięską klasę.
- Dla aᵢ>0 mamy ‖hᵢ‖² ≤ (1−aᵢ)S/aᵢ. Wynika to z Σaᵢhᵢ=0 i nierówności Jensena zastosowanej do wszystkich pozostałych ekspertów. Dowolna różnica dwóch współrzędnych hᵢ jest najwyżej √2‖hᵢ‖. Jeśli kwadrat odstępu dwóch najlepszych średnich logitów przekracza 2(1−a_min)S/a_min, każdy aktywny ekspert indywidualnie zachowuje zwycięzcę. Zachowuje go więc też pełna pula prawdopodobieństw.

W obu warunkach można zastąpić S górnym ograniczeniem. Wektory reszt mają zerową sumę w osi klas i ważoną zerową sumę w osi ekspertów. Przy K ekspertach i C klasach można obliczyć dokładną energię przy pomocy najwyżej (K−1)(C−1) projekcji. W HAR daje to najwyżej 20 projekcji, bez osobnego obliczania gęstej formy kwadratowej 562×562. Jeżeli zostawiamy tylko r głównych kierunków, resztę ograniczamy największą odrzuconą wartością własną razy kwadrat normy wejścia w ustalonych współrzędnych. Buduje to deterministyczną granicę z mniejszą liczbą projekcji, ale może obniżyć pokrycie.

Wdrożenie łączy średnie logity i projekcje w jedno mnożenie macierzy. Jeśli żadna gwarancja nie zachodzi, wykonujemy dokładną arytmetyczną pulę dla niepewnych wierszy. Wynikiem jest klasa top‑1. Odtworzenie pełnego wektora prawdopodobieństw nadal wymaga dodatkowej pracy.

## Porównanie kosztu

Pomiar obejmował pierwotny podział cyfr, nowy podział z seedem 3101 i odłożone osoby HAR. Każdy zbiór miał widoki rozłączne, zachodzące, niemal identyczne i identyczne. Oddzielnie zmierzyliśmy wprowadzanie rekordów pojedynczo, partiami po 64 oraz całą partią testową. Czas każdej metody to mediana 11 przemieszanych powtórzeń po rozgrzaniu, na jednym wątku CPU. Pełna pula wykonuje jeden wspólny GEMM dla logitów wszystkich pięciu ekspertów.

W **31 z 36** układów czasu prosta kontrola jednomyślności ekspertów była szybsza od pełnej puli. Oblicza wszystkie logity jednym GEMM, a prawdopodobieństwa tylko dla wierszy, w których eksperci wskazują różne klasy. Pełny router certyfikacyjny był szybszy od pełnej puli w **11 z 36**, a wersja rzędu 2 w **5 z 36**. Rząd 2 nie pobił kontroli jednomyślności w żadnym z 36 układów. Pełny rząd był szybszy od niej dwukrotnie. Jeden z tych przypadków obejmował pięć identycznych ekspertów; ich prosta deduplikacja dała 0,573 µs na rekord zamiast 1,168 µs routera.

Przy zachodzących widokach HAR pełna gwarancja objęła 65,42% wierszy, a mimo tego router całej partii potrzebował 1,943 µs na rekord wobec 1,618 µs dla pełnej puli i 1,276 µs po kontroli jednomyślności. Przy niemal identycznych widokach rząd 2 gwarantował 98,7% decyzji HAR, lecz nadal przegrywał z pełną pulą. Wysokie pokrycie nie wystarcza, gdy normowanie wejścia, sprawdzenie granicy i obsługa pozostałych wierszy kosztują więcej od pominiętych ekspertów. Liczby dotyczą określonej implementacji NumPy i maszyny; 11 powtórzeń nie odpowiada replikacji na niezależnych urządzeniach.

## Użycie oraz odtworzenie

```python
from okapi.softmax_routing import compile_softmax_router

router = compile_softmax_router(W, biases, weights, center=mean, scale=std, rank=None)
candidate, by_probability, by_unanimity = router.certified(X)
use_candidate = by_probability | by_unanimity
```

Do wdrożenia dokładnej decyzji trzeba dodatkowo uruchomić pełny ensemble na wierszach z use_candidate=False. Przed rozpoczęciem pomiaru zamroziliśmy wcześniejsze C do regresji. Powtórny trening różnił się od wcześniejszych wyników Brier najwyżej o 8,73×10⁻⁷. Eksporty ekspertów, 3663 pomiary czasu, pokrycia i zbiorcze kontrole zgodności decyzji są w katalogu results. Cały benchmark trwał 68,92 s. Następny test bada odmienną możliwość dokładnego zastępowania decyzji: dla dwóch równoważnych binarnych sigmoid o logitach a i b decyzja ich średniej jest dokładnie znakiem a+b.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python fusionbench/examples/real_softmax_routing.py --har har.zip \
  --previous fusionbench/docs/ensemble-study/softmax-certificate/results \
  --out routing-results --export-experts
```

[Przedrejestracja](https://github.com/damiankucharski/fusionbench/blob/0f1d195/docs/ensemble-study/softmax-routing/benchmark-protocol.md) · [Kod OKAPI](https://github.com/damiankucharski/OKAPI/blob/ec94d11/okapi/softmax_routing.py)
