# Projekt: Identyfikacja obiektu FIR za pomocą rekurencyjnej metody najmniejszych kwadratów (RLS) ze współczynnikiem zapominania (Grupa z parametrami $\theta^* = [1,\,9,\,0,\,4]^T$)

## 1. Cel i zakres projektu

Celem projektu jest implementacja, badanie symulacyjne oraz analityczna weryfikacja estymatora **RLS** (*Recursive Least Squares*) służącego do identyfikacji parametrów obiektu dynamicznego typu FIR (skończona odpowiedź impulsowa) na podstawie zakłóconych pomiarów wyjścia.

Projekt składa się z trzech etapów:

1. **Etap 0° – symulacja**: wygenerowanie danych wejście–wyjście `{(u_k, y_k)}` dla zadanych parametrów.
2. **Etap 1° – obiekt stacjonarny** ($\theta^* = \text{const}$): estymacja RLS w wersji podstawowej, badanie zbieżności i wpływu zakłóceń, inicjalizacji i pobudzenia.
3. **Etap 2° – obiekt niestacjonarny** (parametr $b_1^*$ zmienia się skokowo): estymacja RLS ze współczynnikiem zapominania $\lambda$, badanie kompromisu między **szybkością śledzenia zmiany** a **wariancją estymaty** (*tracking–variance tradeoff*).

Projekt koncentruje się na zbadaniu, jak współczynnik zapominania $\lambda \in (0, 1]$ i poziom zakłóceń $\sigma_z$ wpływają na jakość estymacji, oraz na porównaniu wyników symulacji z przybliżonymi zależnościami teoretycznymi.

---

## 2. Model matematyczny

### 2.1. Obiekt

Wyjście obiektu w chwili $k$ jest kombinacją liniową bieżącej i trzech poprzednich wartości wejścia, zakłóconą szumem pomiarowym:

$$
y_k = b_0^* u_k + b_1^* u_{k-1} + b_2^* u_{k-2} + b_3^* u_{k-3} + z_k = \varphi_k^T \theta^* + z_k
$$

gdzie:

$$
\varphi_k = \begin{bmatrix} u_k \\ u_{k-1} \\ u_{k-2} \\ u_{k-3} \end{bmatrix}, \qquad
\theta^* = \begin{bmatrix} b_0^* \\ b_1^* \\ b_2^* \\ b_3^* \end{bmatrix}
$$

Estymator nie zna $\theta^*$. Dysponuje wyłącznie ciągiem $\{(u_k, y_k)\}_{k=1}^{N}$ i ma wyznaczyć estymatę $\hat{\theta}_k$.

### 2.2. Parametry przydzielone grupie

$$
b_0^* = 1, \quad b_1^* = 9, \quad b_2^* = 0, \quad b_3^* = 4 \qquad\Longrightarrow\qquad \theta^* = [1,\; 9,\; 0,\; 4]^T
$$

Uwagi:
- $b_2^* = 0$: estymator nie ma tej informacji. Zbieżność $\hat{b}_2 \to 0$ jest naturalnym testem poprawności.
- Ze względu na $b_2^* = 0$ **nie stosuje się błędów względnych**. Jakość estymacji mierzy się wyłącznie błędem bezwzględnym (sekcja 2.5).
- W etapie 2° zmienia się parametr $b_1^*$ (największy z parametrów).

### 2.3. Wejście i zakłócenie

- **Wejście** $u_k$: ciąg niezależnych zmiennych losowych $\mathcal{N}(0, \sigma_u^2)$, $\sigma_u^2 = 1$ (sygnał pobudzający trwale).
- **Zakłócenie** $z_k$: ciąg niezależnych zmiennych losowych $\mathcal{N}(0, \sigma_z^2)$, niezależny od $u_k$. Spełnione są założenia: $\mathbb{E}[z_k] = 0$, $\text{Var}(z_k) = \sigma_z^2 < \infty$, $z_k$ i $u_k$ niezależne.
- **Warunki początkowe wejścia**: $u_k = 0$ dla $k \le 0$ (zerowe próbki sprzed początku symulacji).

### 2.4. Scenariusze obiektu

**Scenariusz S (stacjonarny, etap 1°):**

$$
\theta_k^* = [1,\, 9,\, 0,\, 4]^T = \text{const}, \quad k = 1, \dots, N
$$

**Scenariusz N (niestacjonarny, etap 2°)** – skokowa zmiana $b_1^*$ w chwili $k_0$:

$$
b_{1,k}^* = \begin{cases} 9 & k < k_0 \\ 5 & k \ge k_0 \end{cases}, \qquad b_0^* = 1,\; b_2^* = 0,\; b_3^* = 4 \;\text{(stałe)}
$$

Wielkość skoku wynosi $\Delta = 5 - 9 = -4$, a $|\Delta| = 4$. W scenariuszu N wektor prawdziwych parametrów zależy od czasu: $\theta_k^* = [1,\, b_{1,k}^*,\, 0,\, 4]^T$.

### 2.5. Miara jakości estymacji

Błąd parametryczny w chwili $k$ (norma euklidesowa):

$$
e_k = \left\| \hat{\theta}_k - \theta_k^* \right\|_2
$$

W badaniach Monte Carlo (sekcja 5.2) używa się błędu średniokwadratowego uśrednionego po realizacjach szumu:

$$
\text{MSE}_k = \frac{1}{M} \sum_{m=1}^{M} \left\| \hat{\theta}_k^{(m)} - \theta_k^* \right\|_2^2, \qquad \text{RMSE}_k = \sqrt{\text{MSE}_k}
$$

---

## 3. Algorytm estymacji RLS

### 3.1. Kryterium

RLS wyznacza rekurencyjnie estymatę minimalizującą ważoną sumę kwadratów błędów predykcji:

$$
\hat{\theta}_k = \arg\min_{\theta} \; \sum_{i=1}^{k} \lambda^{k-i} \left( y_i - \varphi_i^T \theta \right)^2 + \lambda^{k} \, \theta^T P_0^{-1} \theta
$$

Dla $\lambda = 1$ wszystkie próbki mają równą wagę (wersja podstawowa, obiekt stacjonarny). Dla $0 < \lambda < 1$ próbka sprzed $j$ kroków ma wagę $\lambda^j$, więc stare dane są wykładniczo „zapominane" (obiekt niestacjonarny).

### 3.2. Równania rekurencyjne

W każdym kroku $k = 1, \dots, N$:

$$
P_k = \frac{1}{\lambda} \left( P_{k-1} - \frac{P_{k-1} \varphi_k \varphi_k^T P_{k-1}}{\lambda + \varphi_k^T P_{k-1} \varphi_k} \right)
$$

$$
\hat{\theta}_k = \hat{\theta}_{k-1} + P_k \varphi_k \left( y_k - \varphi_k^T \hat{\theta}_{k-1} \right)
$$

Dla $\lambda = 1$ wzór na $P_k$ redukuje się do:

$$
P_k = P_{k-1} - \frac{P_{k-1} \varphi_k \varphi_k^T P_{k-1}}{1 + \varphi_k^T P_{k-1} \varphi_k}
$$

**Jedna implementacja z parametrem $\lambda$ obsługuje oba etapy** ($\lambda = 1$ to wersja podstawowa).

### 3.3. Inicjalizacja

$$
\hat{\theta}_0 = [0,\, 0,\, 0,\, 0]^T, \qquad P_0 = p_0 \cdot I_4, \quad p_0 = 1000 \;\text{(wartość bazowa)}
$$

Duża wartość $p_0$ odpowiada dużej niepewności początkowej. Estymator ufa wtedy pierwszym danym bardziej niż zerowemu punktowi startowemu.

### 3.4. Estymata nierekurencyjna (odniesienie)

Estymata klasycznej metody najmniejszych kwadratów na całym zbiorze danych:

$$
\hat{\theta}_{LS} = (\Phi^T \Phi)^{-1} \Phi^T Y, \qquad \Phi = \begin{bmatrix} \varphi_1^T \\ \vdots \\ \varphi_N^T \end{bmatrix}, \quad Y = \begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix}
$$

---

## 4. Analiza teoretyczna

Poniższe zależności są **przybliżeniami** (dla białego wejścia, $\lambda$ bliskiego 1, stanu ustalonego). Służą do weryfikacji wyników symulacji, a nie jako dokładne wartości.

### 4.1. Obiekt stacjonarny ($\lambda = 1$)

Dla dużego $p_0$ RLS pokrywa się z LS. Estymator jest nieobciążony, a jego macierz kowariancji wynosi $\sigma_z^2 (\Phi^T\Phi)^{-1}$. Ponieważ $\Phi^T \Phi \approx \sigma_u^2 \, k \, I_4$, otrzymuje się:

$$
\mathbb{E}\left\| \hat{\theta}_k - \theta^* \right\|^2 \approx \frac{4 \, \sigma_z^2}{\sigma_u^2 \, k}
$$

**Wnioski:** błąd maleje jak $1/k$ (RMSE jak $1/\sqrt{k}$) i rośnie proporcjonalnie do $\sigma_z^2$.

### 4.2. Obiekt niestacjonarny ($\lambda < 1$) – efektywne okno pamięci

Suma wag próbek wynosi $\sum_{j=0}^{\infty} \lambda^j = \frac{1}{1-\lambda}$, więc estymator „pamięta" w przybliżeniu

$$
N_{\text{eff}} \approx \frac{1}{1 - \lambda} \;\text{próbek}
$$

(np. $\lambda = 0.98 \to 50$, $\lambda = 0.99 \to 100$ próbek).

### 4.3. Wariancja w stanie ustalonym (koszt zapominania)

Dla stałego $\theta^*$ i $\lambda < 1$ kowariancja błędu estymacji w stanie ustalonym wynosi w przybliżeniu $\sigma_z^2 \frac{1-\lambda}{1+\lambda} (\sigma_u^2 I_4)^{-1}$, stąd:

$$
\mathbb{E}\left\| \hat{\theta}_k - \theta^* \right\|^2 \approx \frac{4 \, \sigma_z^2}{\sigma_u^2} \cdot \frac{1 - \lambda}{1 + \lambda}
$$

Wariancja rośnie wraz z oddalaniem się $\lambda$ od 1 (krótsza pamięć to mniej uśredniania szumu) i jest **proporcjonalna do** $\sigma_z^2$.

### 4.4. Odpowiedź na skok parametru (zysk z zapominania)

Po skoku $b_1^*$ o $\Delta$ w chwili $k_0$ wartość oczekiwana błędu estymaty $b_1$ zanika wykładniczo:

$$
\mathbb{E}\left[ \hat{b}_{1,k} \right] - b_{1,k}^* \approx \Delta \cdot \lambda^{\,k - k_0 + 1}, \qquad k \ge k_0
$$

Czas dojścia do pasma $\pm\varepsilon$ wokół nowej wartości:

$$
T_s(\lambda) \approx \frac{\ln\left( \varepsilon / |\Delta| \right)}{\ln \lambda} \;\approx\; \frac{\ln(|\Delta| / \varepsilon)}{1 - \lambda}
$$

Dla $\varepsilon = 0.2$ i $|\Delta| = 4$ (wartości przyjęte w projekcie): $\ln(|\Delta|/\varepsilon) = \ln 20 \approx 3.0$, czyli $T_s \approx 3 N_{\text{eff}}$. Dla $\lambda = 1$ (RLS bez zapominania) zmiana nie zostaje „nadgonionej" (po skoku estymata dąży do średniej ważonej starej i nowej wartości, a macierz $P_k$ maleje do zera, więc algorytm przestaje reagować).

### 4.5. Kompromis dla wyboru $\lambda$

| | $\lambda \to 1$ (długa pamięć) | $\lambda \ll 1$ (krótka pamięć) |
|---|---|---|
| Wariancja w stanie ustalonym (4.3) | mała | duża |
| Czas dojścia po skoku $T_s$ (4.4) | długi | krótki |
| Wrażliwość na zakłócenia $\sigma_z$ | mała | duża |

Nie istnieje $\lambda$ minimalizujące jednocześnie oba kryteria. Optymalny wybór zależy od tego, jak często zmienia się obiekt i jak duży jest szum. Zwiększenie $\sigma_z$ przesuwa najlepszy kompromis w stronę $\lambda$ bliższego 1.

### 4.6. Predykcje teoretyczne dla parametrów projektu

Wartości wyliczone ze wzorów 4.3 i 4.4 ($\sigma_u = 1$, $\varepsilon = 0.2$, $|\Delta| = 4$). Posłużą jako odniesienie w tabeli wyników.

| $\lambda$ | $N_{\text{eff}}$ | $T_s$ (próbki) | RMSE ustalony, $\sigma_z = 0.5$ | RMSE ustalony, $\sigma_z = 2$ |
|---|---|---|---|---|
| 0.90 | 10 | 28 | 0.23 | 0.92 |
| 0.95 | 20 | 58 | 0.16 | 0.64 |
| 0.98 | 50 | 148 | 0.10 | 0.40 |
| 0.99 | 100 | 298 | 0.071 | 0.28 |
| 0.999 | 1000 | ≈ 2994 | wzór 4.3 nie ma zastosowania (stan nieustalony dla $N = 1000$) | wzór 4.3 nie ma zastosowania |

---

## 5. Plan eksperymentów symulacyjnych

### 5.1. Parametry bazowe symulacji

| Parametr | Wartość |
|---|---|
| $\theta^*$ (etap 1°) | $[1, 9, 0, 4]^T$ |
| $b_{1,\text{nowe}}$ (etap 2°) | $5$ |
| Długość przebiegu $N$ | $1000$ |
| Chwila skoku $k_0$ | $500$ |
| Wejście $u_k$ | $\mathcal{N}(0, 1)$ |
| Zakłócenie $z_k$ | $\mathcal{N}(0, \sigma_z^2)$, $\sigma_z \in \{0.5,\; 2\}$ |
| Inicjalizacja | $\hat{\theta}_0 = 0$, $P_0 = p_0 I_4$, $p_0 = 1000$ |
| Wartości $\lambda$ (etap 2°) | $\{1,\; 0.90,\; 0.95,\; 0.98,\; 0.99,\; 0.999\}$ |
| Liczba powtórzeń Monte Carlo | $M = 100$ |
| Generator liczb losowych | `numpy.random.default_rng(seed)`; przebieg pojedynczy: `seed = 0`; Monte Carlo: `seed = 1000 + m`, $m = 1..M$ |

Wszystkie wartości są obowiązkowe. Jeśli którakolwiek zostanie zmieniona, trzeba to wskazać w sprawozdaniu.

### 5.2. Metodologia Monte Carlo

Pojedynczy przebieg jest silnie zaszumiony, dlatego wszystkie krzywe błędów oraz miary liczbowe w tabelach wyznacza się uśredniając po $M$ niezależnych realizacjach $u_k$ i $z_k$ (inne ziarno dla każdej realizacji):

$$
\text{MSE}_k = \frac{1}{M} \sum_{m=1}^{M} \left\| \hat{\theta}_k^{(m)} - \theta_k^* \right\|^2
$$

Wykresy pojedynczych trajektorii $\hat{\theta}_k$ (sekcja 6) wykonuje się dla jednego przebiegu (`seed = 0`).

### 5.3. Etap 0° – Symulacja danych

1. Ustaw $\theta^*$ zgodnie ze scenariuszem (S lub N, sekcja 2.4).
2. Wygeneruj $u_k \sim \mathcal{N}(0,1)$ dla $k = -2, \dots, N$ (z zerowymi wartościami dla $k \le 0$ w regresorze) i $z_k \sim \mathcal{N}(0, \sigma_z^2)$ dla $k = 1..N$.
3. Oblicz $y_k = \varphi_k^T \theta_k^* + z_k$.

**Kryterium ukończenia:** wykres $u_k$, $y_k$ i $y_k - z_k$ (sygnał bez zakłócenia); dwukrotne uruchomienie z tym samym ziarnem daje identyczne dane.

### 5.4. Etap 1° – Obiekt stacjonarny

Wszystkie podpunkty dotyczą scenariusza S i $\lambda = 1$.

**1A. Zbieżność estymat.** Uruchom RLS dla $\sigma_z = 0.5$, $p_0 = 1000$. Zapisz $\hat{\theta}_k$ i $e_k$.

**1B. Porównanie z LS.** Dla $\sigma_z = 0.5$ i $p_0 \in \{10^3, 10^6\}$ porównaj $\hat{\theta}_N$ z $\hat{\theta}_{LS}$ (sekcja 3.4).

**1C. Wpływ zakłócenia.** Wykonaj Monte Carlo dla $\sigma_z \in \{0.5, 2\}$ ($p_0 = 1000$). Porównaj $\text{RMSE}_k$ z teorią $2\sigma_z/(\sigma_u\sqrt{k})$ (wzór 4.1).

**1D. Wpływ inicjalizacji $p_0$.** Wykonaj Monte Carlo dla $p_0 \in \{1,\; 10^3,\; 10^6\}$ ($\sigma_z = 0.5$). Oceń szybkość zbieżności na początku przebiegu.

**1E. Wpływ pobudzenia.** Powtórz 1A dla wejścia stałego $u_k = 1$ ($k \ge 1$). Regresor $\varphi_k$ jest wtedy (od $k \ge 4$) stały, więc dane niosą informację tylko o sumie $b_0 + b_1 + b_2 + b_3 = 14$. Pokaż, że $\sum_i \hat{b}_i \to 14$, ale poszczególne $\hat{b}_i$ **nie zbiegają** do prawdziwych wartości. Pokazuje to konieczność pobudzenia trwałego.

### 5.5. Etap 2° – Obiekt niestacjonarny

Scenariusz N ($b_1^*$: $9 \to 5$ w $k_0 = 500$).

**2A. Brak zapominania.** Uruchom RLS z $\lambda = 1$ ($\sigma_z = 0.5$). Pokaż, że $\hat{b}_1$ nie dochodzi do 5.

**2B. Zapominanie.** Uruchom RLS dla wszystkich $\lambda$ z sekcji 5.1 oraz $\sigma_z \in \{0.5, 2\}$. Zapisz $\hat{\theta}_k$ (pojedynczy przebieg) oraz $\text{MSE}_k$ (Monte Carlo).

**2C. Miary liczbowe** (wyznaczane z uśrednienia Monte Carlo, dla każdej pary $(\lambda, \sigma_z)$):

- **Czas dojścia** $T_s$: najmniejsze $j \ge 0$, dla którego średnia po realizacjach $\overline{\hat{b}_{1,k_0+j}}$ leży w paśmie $|\,\overline{\hat{b}_{1,k_0+j}} - 5\,| \le 0.2$ i pozostaje w nim do końca przebiegu. Jeśli nie następuje to do $k = N$, zapisz „brak".
- **RMSE ustalony przed skokiem**: $\sqrt{\frac{1}{100} \sum_{k=401}^{500} \text{MSE}_k}$. Porównaj z tabelą 4.6 (dla $\lambda \le 0.99$).
- **RMSE całkowity po skoku**: $\sqrt{\frac{1}{N-k_0+1} \sum_{k=k_0}^{N} \text{MSE}_k}$. Miara łączna (szybkość i szum).

Wyniki zestaw w tabeli (wiersze: $\lambda$, kolumny: $T_s$, RMSE ustalony, RMSE całkowity) osobno dla każdego $\sigma_z$, obok wartości teoretycznych z 4.6.

**2D. Wpływ zakłócenia na wybór $\lambda$.** Na podstawie 2C wskaż $\lambda$ minimalizujące RMSE całkowity po skoku osobno dla $\sigma_z = 0.5$ i $\sigma_z = 2$. Zinterpretuj różnicę (sekcja 4.5).

---

## 6. Wymagane wizualizacje

Każdy rysunek zapisz jako plik PNG (min. 150 dpi) w katalogu `results/` z podaną nazwą. Osie muszą być opisane (wielkość i jednostka), z legendą.

| Plik | Zawartość |
|---|---|
| `fig1_dane.png` | $u_k$, $y_k$ oraz $y_k - z_k$ (0°) |
| `fig2_stac_zbieznosc.png` | $\hat{b}_{0..3,k}$ z liniami poziomymi $b_i^*$ (1A) |
| `fig3_stac_blad.png` | $e_k$ (pojedynczy przebieg) oraz $\text{RMSE}_k$ (MC) z krzywą teoretyczną $2\sigma_z/\sqrt{k}$, skala log-log (1A, 1C) |
| `fig4_stac_sigma_p0.png` | $\text{RMSE}_k$ dla $\sigma_z \in \{0.5, 2\}$ oraz dla $p_0 \in \{1, 10^3, 10^6\}$ (1C, 1D) |
| `fig5_pobudzenie.png` | $\hat{b}_{i,k}$ oraz $\sum_i \hat{b}_{i,k}$ dla $u_k = 1$ (1E) |
| `fig6_niestac_lambda1.png` | $\hat{b}_{i,k}$ i prawdziwe $b_{1,k}^*$ (czerwona linia ze skokiem) dla $\lambda = 1$ (2A) |
| `fig7_niestac_lambdy.png` | $\hat{b}_{1,k}$ oraz $b_{1,k}^*$ dla wszystkich $\lambda$ ($\sigma_z = 0.5$ i $\sigma_z = 2$, osobne panele) (2B) |
| `fig8_blad_lambdy.png` | $\text{RMSE}_k$ dla wszystkich $\lambda$ (MC), pionowa linia w $k_0$ (2B) |
| `fig9_kompromis.png` | $T_s$ oraz RMSE ustalony w funkcji $\lambda$ wraz z krzywymi teoretycznymi (4.3, 4.4); ilustracja kompromisu (2C) |

---

## 7. Kryteria akceptacji

Wynik uznaje się za poprawny, jeśli spełnione są wszystkie poniższe warunki:

1. **Zbieżność (1A):** dla $\sigma_z = 0.5$ i $k \ge 300$ wartość $\text{RMSE}_k$ (MC) jest mniejsza niż $0.15$; $\hat{b}_2 \to 0$ (|$\hat{b}_{2,N}$| < 0.1 w pojedynczym przebiegu).
2. **Zgodność z LS (1B):** $\|\hat{\theta}_N - \hat{\theta}_{LS}\|_2 < 10^{-3}$ dla $p_0 \ge 10^3$.
3. **Zgodność z teorią (1C):** $\text{RMSE}_k$ z symulacji różni się od $2\sigma_z/(\sigma_u\sqrt{k})$ nie więcej niż o $30\%$ dla $k \ge 100$.
4. **Pobudzenie (1E):** $|\sum_i \hat{b}_{i,N} - 14| < 0.2$, a co najmniej jeden $\hat{b}_{i,N}$ różni się od $b_i^*$ o więcej niż $0.5$.
5. **Brak zapominania (2A):** dla $\lambda = 1$ wartość $\overline{\hat{b}_{1,N}}$ różni się od $5$ o więcej niż $1$.
6. **Zapominanie (2B, 2C):** dla $\lambda \in \{0.90, 0.95, 0.98, 0.99\}$ czas $T_s$ i RMSE ustalony przed skokiem różnią się od wartości z tabeli 4.6 nie więcej niż o $30\%$. Jeśli różnica jest większa, sprawozdanie musi ją wyjaśnić (np. błąd w implementacji, ograniczenia przybliżeń).
7. **Kompromis (2C, 2D):** widoczna monotoniczność: $T_s$ rośnie, a RMSE ustalony maleje wraz ze wzrostem $\lambda$.
8. **Powtarzalność:** uruchomienie `python src/main.py` odtwarza wszystkie rysunki i tabele bez dodatkowych interwencji.

---

## 8. Implementacja i struktura repozytorium

**Narzędzia:** Python 3.10+, NumPy, Matplotlib (opcjonalnie pandas do tabel).

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── simulate.py     # simulate(theta_fn, N, sigma_z, rng, u=None) -> u, y, theta_true
│   ├── rls.py          # rls(u, y, lam, p0) -> theta_hat (N x 4), P_trace
│   ├── metrics.py      # rmse_curve(), settling_time(), steady_rmse()
│   └── main.py         # eksperymenty 1A-1E, 2A-2D; zapis rysunków i tabel
├── results/            # fig1_*.png ... fig9_*.png, tabele (CSV)
└── report/             # sprawozdanie (PDF)
```

Wymagania implementacyjne:
- Funkcja `rls` ma jeden parametr `lam` (wersja podstawowa dla `lam = 1.0`).
- Wszystkie parametry z sekcji 5.1 zdefiniowane w jednym miejscu (słownik lub stałe na początku `main.py`).
- Do `rls` **nie wolno** przekazywać prawdziwych parametrów $\theta^*$. Estymator dostaje wyłącznie $u$, $y$, `lam`, `p0`.
- Brak zależności od zewnętrznych danych. Wszystkie dane generuje kod.

### Szkic algorytmu RLS (pseudokod)

```
theta = zeros(4);  P = p0 * eye(4)
for k = 1..N:
    phi = [u[k], u[k-1], u[k-2], u[k-3]]          # u[j] = 0 dla j <= 0
    P   = (P - (P @ phi @ phi.T @ P) / (lam + phi.T @ P @ phi)) / lam
    theta = theta + P @ phi * (y[k] - phi.T @ theta)
    zapisz theta jako theta_hat[k]
```

---

## 9. Struktura sprawozdania (PDF)

1. **Wstęp**: cel projektu, opis modelu obiektu, założenia o zakłóceniu i wejściu, przydzielone parametry.
2. **Metoda**: kryterium LS, wyprowadzenie (lub opis) RLS bez i z zapominaniem, inicjalizacja, relacja z LS.
3. **Analiza teoretyczna**: zależności 4.1–4.5, tabela 4.6.
4. **Opis implementacji**: narzędzia, struktura kodu, parametry symulacji (tabela z sekcji 5.1).
5. **Wyniki etapu 1°**: rysunki 1–5, wnioski dla 1A–1E.
6. **Wyniki etapu 2°**: rysunki 6–9, tabela miar 2C (teoria vs. symulacja), wnioski dla 2A–2D.
7. **Wnioski końcowe**: porównanie symulacji z teorią, rekomendacja wyboru $\lambda$ dla obu poziomów zakłóceń, ograniczenia metody.
8. **Dodatek**: kod źródłowy (lub odnośnik do repozytorium).

Wszystkie rysunki i tabele muszą być opisane w tekście i zawierać wniosek (nie wolno ich tylko wstawić bez komentarza).

---

## 10. Status realizacji projektu

- [x] **Etap 0° – Symulacja danych (`src/simulate.py`)**
  - Zaimplementowano generator danych wejście–wyjście dla obiektu FIR 4. rzędu ($y_k = \varphi_k^T \theta_k^* + z_k$) z warunkami początkowymi $u_k = 0$ dla $k \le 0$.
  - Przygotowano definicje scenariuszy parametrów: stacjonarnego (`theta_stationary` dla Etapu 1°) oraz niestacjonarnego ze skokiem $b_1^*: 9 \to 5$ w $k_0 = 500$ (`theta_nonstationary` dla Etapu 2°).
  - Zweryfikowano powtarzalność wyników dla zadanego ziarna generatora `numpy.random.default_rng`.
- [ ] **Etap 1° – Obiekt stacjonarny (`src/rls.py`, eksperymenty 1A–1E)**
- [ ] **Etap 2° – Obiekt niestacjonarny (`src/metrics.py`, eksperymenty 2A–2D)**
- [ ] **Generowanie wykresów i sprawozdanie (`src/main.py`, `results/`, `report/`)**