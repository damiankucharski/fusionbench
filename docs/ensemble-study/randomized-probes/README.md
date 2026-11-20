# Jak ograniczyć fałszywe wykrycia zależności referencji?

Przy maksymalnie 64 rekordach z losowo przydzieloną kolejnością podmiany i zwykłego powtórzenia, test sekwencyjny wskazał bezpośrednio zależną referencję w 91,48% partii. Średnio zużył 30,99 rekordów i obniżył Brier z 0,1446 do 0,1045. W kontroli bez bezpośredniej zależności zadeklarował efekt w 1,94% partii. Heurystyka bez progu zadeklarowała go w 92,08%. Nasze porównanie obejmuje dwanaście nowych seedów i dwa rzeczywiste wyszukiwania GP dla każdego z nich.

## Co daje losowanie kolejności

Na tym samym rekordzie wykonujemy odczyt bazowy, po czym dwa odczyty referencji. Jednemu podajemy zmienioną predykcję wejściową, a drugiemu oryginalną. Niezależny rzut uczciwą monetą ustala kolejność dwóch wywołań. Dla referencji j obserwacja Dₜⱼ wyraża, o ile częściej podmiana zmieniła znak odpowiedzi względem bazowej niż zwykłe powtórzenie. Przyjmuje wartości −1, 0, 1. Warunkowo względem przeszłości i potencjalnych odczytów dla obu pozycji, gdy podmiana nie działa, losowanie kolejności daje zerową wartość oczekiwaną Dₜⱼ. Dopuszcza to różnicę w szumie pierwszego i drugiego wywołania. Potrzebny jest brak efektu przeniesienia między wywołaniami oraz zachowanie warunków randomizacji.

Dla każdej stawki λ obliczamy iloczyn kolejnych czynników 1+λDₜⱼ i uśredniamy całe procesy dla λ ∈ {0,25, 0,50, 0,75, 0,95}. Zatrzymujemy próbę przy pierwszym przekroczeniu progu 40 przez którykolwiek z dwóch obserwatorów. Wskazujemy przekraczającego, gdy jest jeden. Bez rozstrzygnięcia wdrażamy równą mieszaninę drzew. Nierówność Ville’a oraz poprawka na dwóch obserwatorów ograniczają prawdopodobieństwo fałszywego wskazania do 0,05 przy spełnionych założeniach. Dotyczy to także zatrzymywania zależnego od wyników. Zadeklarowana zależność dotyczy odpowiedzi na tę podmianę. Poprawność predykcji jest osobnym przedmiotem oceny.

## Porównanie na odrębnych próbach

Badany kontekst, budżet etykiet źródłowych i mechanizm OKAPI są zgodne z poprzednim eksperymentem. Brak etykiet wdrożeniowych. W głównym teście szum zmienia znak 2% pierwszych i 24% drugich odczytów. Wybrana referencja pomaga zdecydować między drzewami na odrębnych 32 zwykłych przykładach. Jeden rekord interwencyjny wymaga odczytu bazowego, odczytu z podmianą i zwykłego powtórzenia dla każdego z dwóch obserwatorów.

| Przy maksimum 64 rekordów | Brier przy zależności | Prawidłowe wykrycia | Śr. liczba rekordów przy zależności | Fałszywe deklaracje w kontroli |
|---|---:|---:|---:|---:|
| Sekwencyjny e-proces | 0,1045 | 91,48% | 30,99 | 1,94% |
| Dokładny test znaków przy stałej liczbie rekordów | 0,1017 | 96,98% | 64,00 | 2,65% |
| Większa liczba zmian, bez progu dowodowego | 0,1005 | 99,96% | 64,00 | 92,08% |
| Stała mieszanina kandydatów | 0,1446 | 0% | 0 | 0% |

W głównym porównaniu różnica e-proces minus stała mieszanina wyniosła −0,04006 Brier [−0,04160, −0,03852] w sparowanym przedziale t po 12 seedach. Większy budżet pomagał uzyskać rozstrzygnięcie. Przy czterech rekordach sekwencyjny próg pozostał nieosiągnięty w każdej partii. Przy szesnastu wykryto 25,46% bezpośrednich zależności. Przy 64 rekordach średni koszt w kontroli bez efektu wyniósł 63,47.

Jeśli podmiana zawsze wypadła jako druga, oba wywołania różniły się systematycznie szumem. W kontroli bez efektu e-proces wskazał obserwatora w 84,29% partii. Odpowiednia liczba po przywróceniu losowania wyniosła 1,94%. Sam próg statystyczny nie naprawia niekontrolowanego eksperymentu.

## Pliki i zakres wniosków

Kod OKAPI do obliczania sekwencyjnych logarytmów e-wartości i decyzji to `okapi.probes` z gałęzi badawczej, commit `968baa3`. Wyniki symulacji obejmują również stacjonarny szum po 10% na obu pozycjach oraz kontrole z przydziałem nielosowym. Faktyczne obliczenie wygenerowało wszystkie rekordy, aby porównać polityki na identycznych realizacjach. Kolumna `records` przedstawia liczbę odczytanych rekordów dla danej polityki zatrzymania. Dowód dotyczy ostrej hipotezy braku reakcji lub warunkowej niedodatniej średniej różnicy. Klasyfikowanie reagującego obserwatora jako szkodliwego wymaga założenia o przyczynie błędów.

```bash
cd fusionbench
PYTHONPATH=.:../okapi OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m fusionbench.eval.randomized_probes --output docs/ensemble-study/randomized-probes/results
PYTHONPATH=../okapi python -m pytest -q ../okapi/test/test_probes.py
```

Konfiguracja, drzewa, wyniki według seeda i przedziały są w `results/`. Protokół w `protocol.md` opublikowano przed obliczeniami.
