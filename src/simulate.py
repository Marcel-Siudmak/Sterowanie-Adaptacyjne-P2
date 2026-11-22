import numpy as np


def theta_stationary(k: int) -> np.ndarray:
    """Scenariusz S (Etap 1): stałe parametry obiektu."""
    return np.array([1.0, 9.0, 0.0, 4.0])


def theta_nonstationary(k: int, k0: int = 500) -> np.ndarray:
    """Scenariusz N (Etap 2): skokowa zmiana parametru b1 z 9 na 5 w chwili k0."""
    # Uwaga: w README indeksowanie k zaczyna się od 1, a skok jest dla k >= k0 (czyli 500. próbka).
    # W Pythonie (indeksowanie od 0) 500. próbka ma indeks k = 499 (czyli k >= k0 - 1).
    b1 = 9.0 if k < (k0 - 1) else 5.0
    return np.array([1.0, b1, 0.0, 4.0])


def simulate(
    theta_fn,
    N: int,
    sigma_z: float,
    rng: np.random.Generator,
    u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Symuluje obiekt FIR 4. rzędu: y_k = phi_k^T * theta_k^* + z_k.

    Zwraca:
        u: wektor wejść (N,)
        y: wektor zaszumionych wyjść (N,)
        theta_true: macierz prawdziwych parametrów w każdej chwili (N, 4)
    """
    if u is None:
        u = rng.normal(loc=0.0, scale=1.0, size=N)
    else:
        u = np.asarray(u, dtype=float)

    z = rng.normal(loc=0.0, scale=sigma_z, size=N)

    # Doklejamy 3 zera na początek wejścia (warunek u_k = 0 dla k <= 0)
    u_padded = np.concatenate([np.zeros(3), u])

    y = np.zeros(N)
    theta_true = np.zeros((N, 4))

    for k in range(N):
        # Wektor regresorów phi_k = [u_k, u_{k-1}, u_{k-2}, u_{k-3}]
        # Dzięki doklejeniu 3 zer, w u_padded próbka u_k jest pod indeksem k + 3
        phi_k = np.array([
            u_padded[k + 3],
            u_padded[k + 2],
            u_padded[k + 1],
            u_padded[k],
        ])

        theta_k = theta_fn(k)
        theta_true[k] = theta_k
        y[k] = phi_k @ theta_k + z[k]

    return u, y, theta_true