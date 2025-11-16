# GKOM 2025Z | Motion Blur

## Terminy
Projekt oddawany jest w 2 etapach:
* 1. prezentacja wstępnego programu np. implementacja okna, cieniowania i
wczytywania obiektów
* 2. prezentacja ostatecznej wersji programu.

Konieczne jest zaprezentowanie projektu odpowiedniemu prowadzącemu 2
razy - brak pierwszej prezentacji uniemożliwia oddanie projektu.

* Oddanie pierwszego etapu projektu 01.12.2025
* Oddanie ostatecznej wersji programu 23.01.2026

## Motion blur
### Przypisanie zadań gdzie:
* Weronika Maślana (W)
* Zosia Jasina (Z)
* Maciek Kaniewski (M)

### Instrukcja
* C - zmień kamerę
* WASD - poruszaj samochodem

### Prowadzący: dr inż. Łukasz Dąbała
W ramach projektu należy stworzyć prostą grę wyścigową, w której dodany
zostanie motion-blur
* 1. wsparcie dla różnego typu plików wejściowych z modelem 3D: obj. Intere-
suje nas wczytanie trasy, modelu samochodu oraz jego wnętrza. (M)
* 2. cieniowanie Phonga (W)
* 3. interpretację właściwości materiału: diffuse, specular (W)
* 4. wsparcie dla tekstur typu diffuse, specular (M)
* 5. wsparcie dla świateł punktowych - wystarczy pojedyncze światło (Z)
* 6. kamerę perspektywiczną (Z)
** (a) przeczepioną do wnętrza samochodu (możliwość kierowania z pierw-
szej osoby)
** (b) przyczepioną nad tyłem samochodu (możliwość kierowania z trzeciej
osoby)
* 7. efekt motion-blur (W)
**(a) obiekty powinny rozmazywać się wraz z ruchem
**(b) należy uwzględnić odległość od obiektów - obiekty bliższe rozmazują
się bardziej niż dalsze

* 8. kolizje - z objektami ze sceny
* 9. tor - owalny ze startem/metą oraz ograniczeniem by nie dało się wyjechać poza trasę
* 10. (opcjonalnie) timer - mierzymy czas przejechania trasy
* 11. (opcjonalnie) dodanie przeszkód (np. drzew) na trasie

## Wymagania projektu
* W ramach projektu należy stworzyć program, który będzie realizował opisane w
temacie funkcje. Projekt jest zadaniem zespołowym, gdzie każdy zespół składa
się z 3 osób.
* Głównym językiem programowania może być język Python lub C++. Do
realizacji funkcji graficznych należy wykorzystać bibliotekę OpenGL wraz z ję-
zykiem shaderów GLSL.
* Za projekt można uzyskać maksymalnie x × 15p., gdzie x to liczba osób w
zespole. Każdy z członków zespołu może dostać maksymalnie 15 punktów.
## Ocenie w ramach projektu podlegają:
* 1. Działanie programu - realizacja funkcji (9 p.)
* 2. Efekty wizualne - prezentacja działania programu oraz kroku algorytmu
w przyjemnie wizualny sposób (przygotowanie modeli, scenerii itd.) (2 p.)
* 3. Jakość kodu (3 p.)
* 4. Prezentacja wykonana na ostatnim wykładzie (1 p.)
Projekt musi być pokazany odpowiedniemu prowadzącemu przed terminem
ostatniego wykładu. Dodatkowo, brak prezentacji na ostatnim wykładzie skut-
kuje niezaliczeniem projektu.
