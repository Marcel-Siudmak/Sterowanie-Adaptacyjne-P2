import numpy as np


def mse_curve(theta_hats: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """
    Oblicza krzywą błędu średniokwadratowego MSE_k z M realizacji Monte Carlo (sekcja 2.5 i 5.2).

    Parametry:
        theta_hats: tablica o wymiarze (M, N, 4) z estymatami dla M realizacji
        theta_true: tablica o wymiarze (N, 4) z prawdziwymi parametrami
    Zwraca:
        mse: wektor o długości N z wartościami MSE_k
    """
    # Różnica (M, N, 4) -> kwadrat normy euklidesowej po osi parametrów -> średnia po M realizacjach
    sq_errors = np.sum((theta_hats - theta_true[None, :, :]) ** 2, axis=2)
    return np.mean(sq_errors, axis=0)


def rmse_curve(theta_hats: np.ndarray, theta_true: np.ndarray) -> np.ndarray:
    """Oblicza pierwiastek z MSE_k (RMSE_k) w każdym kroku k."""
    return np.sqrt(mse_curve(theta_hats, theta_true))


def settling_time(
    b1_mean: np.ndarray,
    target: float = 5.0,
    eps: float = 0.2,
    k0: int = 500,
) -> int | None:
    """
    Wyznacza czas dojścia T_s po skoku w chwili k0 (sekcja 5.5, punkt 2C).
    Szuka najmniejszego j >= 0, dla którego |b1_mean[k0 - 1 + j] - target| <= eps
    i warunek ten jest spełniony aż do końca przebiegu.
    """
    start_idx = k0 - 1  # 500. próbka w indeksowaniu od 0 to indeks 499
    post_jump_errors = np.abs(b1_mean[start_idx:] - target)
    n_post = len(post_jump_errors)

    for j in range(n_post):
        if np.all(post_jump_errors[j:] <= eps):
            return int(j)

    return None


def steady_rmse(mse: np.ndarray, k_start: int, k_end: int) -> float:
    """
    Oblicza uśrednione RMSE na przedziale kroków [k_start, k_end] (włącznie, numeracja od 1).
    Używane do wyznaczania RMSE ustalonego przed skokiem (401..500) i całkowitego po skoku (500..1000).
    """
    segment = mse[k_start - 1 : k_end]
    return float(np.sqrt(np.mean(segment)))