import numpy as np


def rls(
    u: np.ndarray,
    y: np.ndarray,
    lam: float = 1.0,
    p0: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rekurencyjna metoda najmniejszych kwadratów (RLS) ze współczynnikiem zapominania.

    Parametry:
        u: wektor wejść (N,)
        y: wektor wyjść (N,)
        lam: współczynnik zapominania lambda w (0, 1] (dla lam=1.0 wersja podstawowa)
        p0: początkowa wartość na przekątnej macierzy kowariancji P0 = p0 * I_4

    Zwraca:
        theta_hat: macierz estymat parametrów w kolejnych krokach (N, 4)
        P_trace: wektor śladu macierzy P_k w kolejnych krokach (N,)
    """
    u = np.asarray(u, dtype=float)
    y = np.asarray(y, dtype=float)
    N = len(u)

    # Inicjalizacja (sekcja 3.3 README): startujemy od zer i dużej niepewności
    theta = np.zeros(4, dtype=float)
    P = p0 * np.eye(4, dtype=float)

    # Warunki początkowe wejścia: u_k = 0 dla k <= 0
    u_padded = np.concatenate([np.zeros(3), u])

    theta_hat = np.zeros((N, 4), dtype=float)
    P_trace = np.zeros(N, dtype=float)

    for k in range(N):
        # Wektor kolumnowy phi_k o wymiarze (4, 1)
        phi = np.array([
            u_padded[k + 3],
            u_padded[k + 2],
            u_padded[k + 1],
            u_padded[k],
        ], dtype=float).reshape(4, 1)

        # Mianownik we wzorze na P_k: lam + phi^T * P * phi (skalar)
        denom = lam + float((phi.T @ P @ phi)[0, 0])

        # Aktualizacja macierzy kowariancji P_k (sekcja 3.2 README)
        P = (P - (P @ phi @ phi.T @ P) / denom) / lam

        # Błąd predykcji: y_k - phi_k^T * theta_{k-1}
        pred_error = y[k] - float((phi.T @ theta.reshape(4, 1))[0, 0])

        # Aktualizacja wektora estymat theta_k
        theta = theta + (P @ phi).ravel() * pred_error

        theta_hat[k] = theta
        P_trace[k] = np.trace(P)

    return theta_hat, P_trace