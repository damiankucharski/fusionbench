# Czy jedna sigmoida zachowuje korzyść zespołu na prawdziwych cechach?

**Podstawowy test na Spambase nie wykazał przewagi arytmetycznej puli sześciu ekspertów z rozłącznymi wejściami.** Na 921 odłożonych wiadomościach pula uzyskała Brier 0,10304, wybrany na walidacji najlepszy ekspert 0,10287, a jedna regresja logistyczna na wszystkich 57 cechach 0,05804. Pojedynczy model, któremu walidacja pozwalała także wybrać log1p cech, osiągnął 0,04358. Niższy Brier oznacza lepsze prawdopodobieństwa. Żadna postać kompresji tej puli nie spełniła więc warunku zachowania potwierdzonej przewagi nad oboma punktami odniesienia.

Na UCI HAR rozłączne modele rozpoznawały kierunek chodzenia po schodach. Pula poprawiła Brier wobec najlepszego eksperta z 0,04781 do 0,02683, z 95-procentowym przedziałem poprawy od 0,00417 do 0,04215. Jedna logistyka z dostępem do wszystkich cech tej samej osoby osiągnęła jeszcze niższy Brier 0,01903.

## Co robią komplementarne i pokrywające się widoki

Sześciu ekspertów w wariancie rozłącznym otrzymuje po jednej z sześciu części pełnych cech. W wariancie zachodzącym każdy otrzymuje pięć z sześciu części. Pełna logistyka korzysta ze wszystkich wejść i tego samego budżetu etykiet. Walidacja wybiera regularyzację oraz najlepszego członka. Pula zawsze zwraca średnią sześciu prawdopodobieństw. Kompilacja stycznej wykorzystuje wartość i pochodną puli w średnim wejściu treningowym. Wariant sieczny uwzględnia także jej odpowiedzi w dwóch punktach wyznaczonych przez kowariancję cech. Obie metody korzystają z parametrów ekspertów. Odrębne odniesienie destylacyjne uczy się z ich miękkich prognoz na wejściach treningowych.

| Zbiór i widoki | Najlepszy członek | Arytmetyczna pula | Średnie parametry, jedna sigmoida | Styczna | Sieczna | Destylacja | Pełna logistyka |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spambase, rozłączne | 0,10287 | 0,10304 | 0,07188 | 0,07648 | 0,12283 | 0,11882 | **0,05804** |
| Spambase, zachodzące | 0,05484 | 0,05304 | **0,05238** | 0,07210 | 0,08867 | 0,05794 | 0,05804 |
| HAR, rozłączne | 0,04781 | 0,02683 | 0,01997 | 0,02040 | 0,02576 | 0,02723 | **0,01903** |
| HAR, zachodzące | 0,02130 | 0,02024 | 0,02008 | 0,02008 | 0,02008 | 0,02027 | **0,01903** |

Dla rozłącznego Spambase przedział różnicy najlepszy członek minus pula wynosił od −0,00789 do 0,00866, a pełna logistyka minus pula od −0,05149 do −0,03826. Styczna osiągnęła Brier o 0,02655 niższy od puli, z przedziałem od −0,03094 do −0,02166 dla różnicy styczna minus pula. Spełnia to próg zachowania jakości zadaniowej +0,002 względem tej puli, chociaż RMSE różnicy prawdopodobieństw wynosi aż 0,13358. Wierne prawdopodobieństwa i dobry wynik zadania to dwa różne kryteria.

W dodatkowym wariancie zachodzącym Spambase pula miała o 0,00501 niższy Brier od pełnej liniowej logistyki, z przedziałem od 0,00251 do 0,00792. Przewaga nad najlepszym składnikiem była niepewna, z przedziałem od −0,00072 do +0,00477, a pełna logistyka z transformacją log1p ponownie była lepsza, 0,04358. Średnie parametry ekspertów zachowały Brier zachodzącej puli w granicy +0,002 (górny kraniec +0,00025). Styczna ten warunek przekroczyła (górny kraniec +0,02558). W kontrolach z sześcioma identycznymi ekspertami styczna, sieczna i średnie parametry odtworzyły pulę z dokładnością numeryczną.

## Kiedy średnie wagi bezpiecznie zastępują pulę?

W konkretnym przykładzie można policzyć logity sześciu ekspertów lᵢ i ich wariancję. Maksymalna bezwzględna druga pochodna sigmoidy wynosi 1/(6√3). W rozwinięciu Taylora wokół średniego logitu liniowe składniki po uśrednieniu się znoszą. Stąd dla dowolnego wejścia

```text
|meanᵢ σ(lᵢ) − σ(meanᵢ lᵢ)| ≤ Varᵢ(lᵢ)/(12√3).
```

Ten certyfikat dotyczy jednej sigmoidy otrzymanej przez uśrednienie wag i biasów. Może wymagać obliczenia logitów lub uprzednio wyznaczonej kwadratowej formy ich wariancji. W eksperymencie mierzymy pokrycie, bez przyjmowania zaoszczędzonego czasu.

| Zbiór i widoki | Certyfikowane przy błędzie ≤0,05 | Rzeczywiście w granicy ≤0,05 | Największy błąd w certyfikowanym podzbiorze |
|---|---:|---:|---:|
| Spambase, rozłączne | 26,82% | 38,11% | 0,0412 |
| Spambase, zachodzące | 41,15% | 94,46% | 0,0356 |
| HAR, rozłączne | 2,71% | 66,10% | 0,0359 |
| HAR, zachodzące | 100,00% | 100,00% | 0,0082 |

Nie wystąpiło żadne naruszenie ograniczenia na obserwowanych wejściach. Jego zachowawczość zależała od rozrzutu logitów. Dla obu kontroli z identycznymi ekspertami pokrycie wyniosło 100%.

## Dane, granice wnioskowania i odtworzenie

W Spambase wykorzystujemy 57 opracowanych cech wiadomości. W HAR korzystamy z 561 gotowych cech okien pomiarów inercyjnych. Eksportowane sigmoidy działają bez składowych ekspertów na tych samych liczbowych wejściach. Standaryzację wchłonięto w wagi i bias. Do nauki, walidacji i testu Spambase trafiło odpowiednio 2759, 921 i 921 wiadomości. Identyczne wektory cech są zawsze w jednej części. 921 wiadomości testowych odpowiada 826 unikatowym wektorom. Brak identyfikatorów nadawcy ogranicza wnioskowanie o przesunięciu do innych nadawców.

W HAR podział obejmuje odpowiednio 17, 4 i 9 rozłącznych osób oraz 1687, 372 i 891 okien. Wynik HAR daje równą wagę każdej osobie. Użyliśmy 4000 sparowanych powtórzeń bootstrapu całych grup duplikatów Spambase lub całych uczestników HAR. Przedziały szacują niepewność testu dla jednego ustalonego treningu. Dziewięć odłożonych osób ogranicza precyzję replikacji. Silny wynik na rzeczywistych zespołach nieliniowych pozostaje otwarty.

Obliczenia na lokalnym CPU trwały 9,24 s. Pliki methods.csv, contrasts.csv i certificates.csv zawierają wszystkie porównania. test_groups.csv.gz umożliwia odtworzenie przedziałów. raw_models.json.gz zawiera 24 wyeksportowane modele. Ich prognozy zweryfikowano na wszystkich wejściach testowych po przeliczeniu wag do oryginalnych jednostek.

```bash
PYTHONPATH="$PWD/fusionbench:$PWD/okapi" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python fusionbench/examples/real_sigmoid_pool.py --spambase spambase.zip --har har.zip --out real-pool-results
```

[Protokół sprzed obliczeń](https://github.com/damiankucharski/fusionbench/blob/e7904d9/docs/ensemble-study/real-sigmoid-pool/protocol.md) · [UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) · [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
