# Jedna reguła decyzji lub dwie skrajne reguły dla ciągłej rodziny zespołów

Dwa głosy i płaszczyznę trzeciego znamy. Nieznany kąt trzeciego głosu może przyjąć dowolną wartość od 0° do 60°. Bezszumowe końcowe etykiety pozostawiają przedział zgodnych kątów [L,U]. **Przy izotropowym teście jedna sieć z kątem środkowym minimalizuje najgorszy błąd klasyfikacji. Dla Brier dokładne rozwiązanie minimaksowe stanowi średnia dwóch reguł krańcowych.** Liczba pozostałych możliwych reguł może być nieskończona. [Wyprowadzenie](derivation.md)

Jeśli kąty mierzymy w radianach, D = (U−L)/(2π) jest dokładnie odsetkiem rozkładu testowego, dla którego nadal możliwe są dwie różne etykiety. Kąt środkowy daje najgorszy błąd D/2. Cztery współdzielone neurony progowe dla dwóch stałych i dwóch skrajnych głosów wystarczają, by uzyskać minimaksowy Brier D/4. Na wejściach spornych przewidują 0,5. Przy jednostajnym priorze kąta posterior ma niższy średni Brier D/6, ale większy Brier D/3 dla każdego końca przedziału. Kryterium decyzji ma więc znaczenie dla wyboru architektury.

### Różne tempa graniczne mogą dawać ten sam wynik przy realnym budżecie

Zakładamy trening z kierunkiem dominującym o wielkości chi z k=15 stopniami swobody. Całkowanie dokładnego prawdopodobieństwa zgodności daje oczekiwany odsetek niejednoznacznych wejść testowych.

| Etykiety treningowe | Prawdziwy kąt 0° | Prawdziwy kąt 30° |
|---:|---:|---:|
| 512 | 13,761% | 13,759% |
| 8192 | 11,003% | 10,939% |
| 1 000 000 | 7,716% | 1,982% |

Przy 512 etykietach podstawowe porównanie nie wykazało praktycznej różnicy. Symulacje na surowych danych dały odpowiednio 13,84% i 13,76% oczekiwanej niejednoznaczności po dopasowaniu. Na brzegu przestrzeni (0°) oczekiwana szerokość zgodnego przedziału maleje asymptotycznie jak n⁻¹⁄ᵏ, a dla każdego ustalonego kąta wewnętrznego jak n⁻¹. Przy k=15 asymptotyka lokalna dla 30° nie jest trafnym przybliżeniem przy 512 ani 8192 etykietach.

W osobnym warunku z kątem losowanym jednostajnie i 512 etykietami średnia niejednoznaczność wyniosła 11,38%. Minimax uzyskał Brier 0,02837 dla prawdziwych kątów, posterior 0,02004. Przy ocenianiu tych samych predykcji względem obu dopuszczalnych skrajności wyniki wyniosły 0,02837 i 0,03782. Wszystkie 18 432 dopasowania przedziału zawierały prawdziwy kąt (z tolerancją numeryczną 10⁻¹²). Symulowaliśmy 16 niezależnych zestawów po 32 próby na warunek. Wyniki dla czterech budżetów wewnątrz próby używają zagnieżdżonych obserwacji.

Ten przypadek wymaga znajomości dwóch rzeczywistych głosów, płaszczyzny i monotonii zmian decyzji wzdłuż przedziału. W innych rodzinach trzeba oddzielnie sprawdzić, czy dwie dopuszczalne funkcje skrajne wyznaczają wszystkie sporne prognozy.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.majority_interval --output /tmp/majority-interval
```

[Protokół](protocol.md) · [Dokładne całki i asymptotyki](results/exact.csv) · [Surowe próby](results/trials.csv.gz) · [Średnie](results/means.csv) · [Czas](results/timing.json)
