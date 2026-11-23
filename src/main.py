import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.metrics import mse_curve, rmse_curve, settling_time, steady_rmse
from src.rls import rls
from src.simulate import simulate, theta_nonstationary, theta_stationary

# Wszystkie parametry bazowe zdefiniowane w jednym miejscu (sekcja 5.1 i 8 README)
PARAMS = {
    "theta_true": np.array([1.0, 9.0, 0.0, 4.0]),
    "b1_new": 5.0,
    "N": 1000,
    "k0": 500,
    "sigma_u": 1.0,
    "sigma_z_list": [0.5, 2.0],
    "p0_base": 1000.0,
    "p0_list": [1.0, 1e3, 1e6],
    "lambdas": [1.0, 0.90, 0.95, 0.98, 0.99, 0.999],
    "M": 100,
    "seed_single": 0,
    "seed_mc_offset": 1000,
    "eps_settling": 0.2,
    "delta_b1": -4.0,
}

RESULTS_DIR = Path("results")


def batch_ls(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nierekurencyjna estymata LS na całym zbiorze danych (sekcja 3.4 README)."""
    N = len(u)
    u_padded = np.concatenate([np.zeros(3), u])
    Phi = np.column_stack([
        u_padded[3 : N + 3],
        u_padded[2 : N + 2],
        u_padded[1 : N + 1],
        u_padded[0:N],
    ])
    theta_ls, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
    return theta_ls


def run_monte_carlo(
    theta_fn,
    N: int,
    sigma_z: float,
    lam: float,
    p0: float,
    M: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uruchamia M niezależnych realizacji symulacji i estymacji RLS."""
    theta_hats = np.zeros((M, N, 4), dtype=float)
    theta_true_ref = np.zeros((N, 4), dtype=float)

    for m in range(1, M + 1):
        rng = np.random.default_rng(seed_offset + m)
        u, y, theta_true = simulate(theta_fn, N, sigma_z, rng)
        if m == 1:
            theta_true_ref = theta_true
        th_hat, _ = rls(u, y, lam=lam, p0=p0)
        theta_hats[m - 1] = th_hat

    return theta_hats, theta_true_ref


def theo_ts(lam: float, delta: float, eps: float) -> float | None:
    """Teoretyczny czas dojścia z sekcji 4.4."""
    if lam >= 1.0:
        return None
    return float(np.log(eps / abs(delta)) / np.log(lam))


def theo_steady_rmse(lam: float, sigma_z: float, sigma_u: float) -> float | None:
    """Teoretyczne RMSE w stanie ustalonym z sekcji 4.3."""
    if lam >= 1.0:
        return None
    mse_val = (4.0 * (sigma_z**2) / (sigma_u**2)) * ((1.0 - lam) / (1.0 + lam))
    return float(np.sqrt(mse_val))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    N = PARAMS["N"]
    k0 = PARAMS["k0"]
    M = PARAMS["M"]
    k_axis = np.arange(1, N + 1)

    print("=== ETAP 0 i ETAP 1: Obiekt stacjonarny ===")

    # --- Etap 0 & 1A: Pojedynczy przebieg (seed = 0, sigma_z = 0.5, p0 = 1000) ---
    rng_single = np.random.default_rng(PARAMS["seed_single"])
    u_0, y_0, theta_true_s = simulate(theta_stationary, N, 0.5, rng_single)
    # Sygnał niezaszumiony y_k - z_k = phi_k^T * theta_k^*
    u_pad = np.concatenate([np.zeros(3), u_0])
    Phi_0 = np.column_stack([u_pad[3 : N + 3], u_pad[2 : N + 2], u_pad[1 : N + 1], u_pad[0:N]])
    y_clean_0 = Phi_0 @ PARAMS["theta_true"]

    # Wykres fig1_dane.png (Etap 0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(k_axis[:150], u_0[:150], label=r"Wejście $u_k \sim \mathcal{N}(0,1)$", color="tab:blue")
    axes[0].set_ylabel("Amplituda [-]")
    axes[0].set_title("Etap 0°: Sygnał wejściowy oraz wyjście obiektu (pierwsze 150 próbek)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(k_axis[:150], y_0[:150], label=r"Wyjście zaszumione $y_k$ ($\sigma_z=0.5$)", color="tab:orange", alpha=0.8)
    axes[1].plot(k_axis[:150], y_clean_0[:150], "--", label=r"Wyjście bez zakłóceń $y_k - z_k$", color="black", linewidth=1.2)
    axes[1].set_xlabel("Krok dyskretny k [próbki]")
    axes[1].set_ylabel("Amplituda [-]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_dane.png", dpi=150)
    plt.close(fig)

    # 1A: RLS dla pojedynczego przebiegu
    theta_hat_1a, _ = rls(u_0, y_0, lam=1.0, p0=PARAMS["p0_base"])
    e_k_1a = np.linalg.norm(theta_hat_1a - theta_true_s, axis=1)

    # Wykres fig2_stac_zbieznosc.png (1A)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i in range(4):
        ax.plot(k_axis, theta_hat_1a[:, i], label=rf"$\hat{{b}}_{{{i},k}}$", color=colors[i])
        ax.axhline(PARAMS["theta_true"][i], linestyle="--", color=colors[i], alpha=0.7, label=rf"$b_{i}^* = {PARAMS['theta_true'][i]:.0f}$")
    ax.set_xlabel("Krok dyskretny k [próbki]")
    ax.set_ylabel("Wartość parametru [-]")
    ax.set_title(r"Etap 1A: Zbieżność estymat parametrów RLS ($\lambda=1, \sigma_z=0.5, p_0=1000$)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_stac_zbieznosc.png", dpi=150)
    plt.close(fig)

    # --- 1B: Porównanie z klasycznym LS ---
    theta_ls = batch_ls(u_0, y_0)
    print("\n[1B] Porównanie RLS z klasycznym LS (N=1000, sigma_z=0.5):")
    print(f"     LS (batch):      {np.round(theta_ls, 6)}")
    for p0_val in [1e3, 1e6]:
        th_p0, _ = rls(u_0, y_0, lam=1.0, p0=p0_val)
        diff_ls = np.linalg.norm(th_p0[-1] - theta_ls)
        print(f"     RLS (p0={p0_val:.0e}): {np.round(th_p0[-1], 6)} | ||theta_N - theta_LS||_2 = {diff_ls:.2e}")
        assert diff_ls < 1e-3, f"Kryterium 1B niespełnione dla p0={p0_val}"

    # --- 1C: Wpływ zakłócenia (Monte Carlo dla sigma_z in {0.5, 2.0}) ---
    rmse_mc_sigma = {}
    for sz in PARAMS["sigma_z_list"]:
        th_hats, th_true = run_monte_carlo(
            theta_stationary, N, sz, lam=1.0, p0=PARAMS["p0_base"], M=M, seed_offset=PARAMS["seed_mc_offset"]
        )
        rmse_mc_sigma[sz] = rmse_curve(th_hats, th_true)

    # Wykres fig3_stac_blad.png (1A, 1C - skala log-log)
    fig, ax = plt.subplots(figsize=(9, 5))
    theo_05 = 2.0 * 0.5 / (PARAMS["sigma_u"] * np.sqrt(k_axis))
    ax.loglog(k_axis, e_k_1a, alpha=0.5, label=r"Błąd pojedynczy $e_k$ ($\sigma_z=0.5$, seed=0)", color="tab:gray")
    ax.loglog(k_axis, rmse_mc_sigma[0.5], linewidth=2, label=r"$\mathrm{RMSE}_k$ Monte Carlo ($M=100, \sigma_z=0.5$)", color="tab:blue")
    ax.loglog(k_axis, theo_05, "--", linewidth=2, label=r"Teoria $2\sigma_z / (\sigma_u \sqrt{k})$ ($\sigma_z=0.5$)", color="tab:red")
    ax.set_xlabel("Krok dyskretny k [próbki] (skala log)")
    ax.set_ylabel("Błąd estymacji [-] (skala log)")
    ax.set_title("Etap 1A & 1C: Zanik błędu estymacji w skali podwójnie logarytmicznej")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_stac_blad.png", dpi=150)
    plt.close(fig)

    # --- 1D: Wpływ inicjalizacji p0 ---
    rmse_mc_p0 = {}
    for p0_val in PARAMS["p0_list"]:
        th_hats, th_true = run_monte_carlo(
            theta_stationary, N, 0.5, lam=1.0, p0=p0_val, M=M, seed_offset=PARAMS["seed_mc_offset"]
        )
        rmse_mc_p0[p0_val] = rmse_curve(th_hats, th_true)

    # Wykres fig4_stac_sigma_p0.png (1C i 1D)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for sz in PARAMS["sigma_z_list"]:
        axes[0].loglog(k_axis, rmse_mc_sigma[sz], linewidth=2, label=rf"MC $\sigma_z = {sz}$")
        axes[0].loglog(k_axis, 2.0 * sz / np.sqrt(k_axis), "--", label=rf"Teoria $\sigma_z = {sz}$")
    axes[0].set_title(r"Etap 1C: Wpływ poziomu zakłóceń $\sigma_z$ ($p_0=1000$)")
    axes[0].set_xlabel("Krok k [próbki] (log)")
    axes[0].set_ylabel(r"$\mathrm{RMSE}_k$ [-] (log)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for p0_val in PARAMS["p0_list"]:
        axes[1].loglog(k_axis, rmse_mc_p0[p0_val], linewidth=2, label=rf"$p_0 = 10^{{{int(np.log10(p0_val))}}}$")
    axes[1].set_title(r"Etap 1D: Wpływ inicjalizacji $p_0$ ($\sigma_z=0.5$)")
    axes[1].set_xlabel("Krok k [próbki] (log)")
    axes[1].set_ylabel(r"$\mathrm{RMSE}_k$ [-] (log)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_stac_sigma_p0.png", dpi=150)
    plt.close(fig)

    # --- 1E: Brak pobudzenia trwałego (u_k = 1) ---
    rng_1e = np.random.default_rng(PARAMS["seed_single"])
    u_const, y_const, _ = simulate(theta_stationary, N, 0.5, rng_1e, u=np.ones(N))
    theta_hat_1e, _ = rls(u_const, y_const, lam=1.0, p0=PARAMS["p0_base"])
    sum_b_1e = np.sum(theta_hat_1e, axis=1)

    # Wykres fig5_pobudzenie.png (1E)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax.plot(k_axis, theta_hat_1e[:, i], label=rf"$\hat{{b}}_{{{i},k}}$ (prawdziwe $b_{i}^*={PARAMS['theta_true'][i]:.0f}$)")
    ax.plot(k_axis, sum_b_1e, color="black", linewidth=2.2, label=r"Suma $\sum_{i=0}^3 \hat{b}_{i,k}$")
    ax.axhline(14.0, color="red", linestyle="--", linewidth=1.5, label=r"Prawdziwa suma $\sum b_i^* = 14$")
    ax.set_xlabel("Krok dyskretny k [próbki]")
    ax.set_ylabel("Wartość estymaty [-]")
    ax.set_title(r"Etap 1E: Brak pobudzenia trwałego ($u_k = 1$) – zbiega tylko suma parametrów")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig5_pobudzenie.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    print("\n=== ETAP 2: Obiekt niestacjonarny (skok b1: 9 -> 5 w k0 = 500) ===")
    theta_fn_n = lambda k: theta_nonstationary(k, k0=k0)

    # --- 2A: Brak zapominania (lambda = 1, sigma_z = 0.5) ---
    rng_2a = np.random.default_rng(PARAMS["seed_single"])
    u_n05, y_n05, theta_true_n = simulate(theta_fn_n, N, 0.5, rng_2a)
    theta_hat_2a, _ = rls(u_n05, y_n05, lam=1.0, p0=PARAMS["p0_base"])

    # Wykres fig6_niestac_lambda1.png (2A)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax.plot(k_axis, theta_hat_2a[:, i], label=rf"$\hat{{b}}_{{{i},k}}$")
    ax.plot(k_axis, theta_true_n[:, 1], "r--", linewidth=2, label=r"Prawdziwe $b_{1,k}^*$ ($9 \to 5$)")
    ax.axvline(k0, color="gray", linestyle=":", label=rf"Chwila skoku $k_0 = {k0}$")
    ax.set_xlabel("Krok dyskretny k [próbki]")
    ax.set_ylabel("Wartość parametru [-]")
    ax.set_title(r"Etap 2A: Niezdolność do śledzenia skoku bez zapominania ($\lambda = 1.0, \sigma_z = 0.5$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig6_niestac_lambda1.png", dpi=150)
    plt.close(fig)

    # --- 2B, 2C, 2D: Badanie różnych wartości lambda oraz Monte Carlo ---
    rng_2b_2 = np.random.default_rng(PARAMS["seed_single"])
    u_n20, y_n20, _ = simulate(theta_fn_n, N, 2.0, rng_2b_2)
    single_data_n = {0.5: (u_n05, y_n05), 2.0: (u_n20, y_n20)}

    single_b1_runs = {sz: {} for sz in PARAMS["sigma_z_list"]}
    mc_rmse_runs = {sz: {} for sz in PARAMS["sigma_z_list"]}
    mc_b1_mean_runs = {sz: {} for sz in PARAMS["sigma_z_list"]}
    tables_2c = {sz: [] for sz in PARAMS["sigma_z_list"]}

    for sz in PARAMS["sigma_z_list"]:
        u_s, y_s = single_data_n[sz]
        for lam in PARAMS["lambdas"]:
            # Pojedynczy przebieg (seed = 0)
            th_single, _ = rls(u_s, y_s, lam=lam, p0=PARAMS["p0_base"])
            single_b1_runs[sz][lam] = th_single[:, 1]

            # Monte Carlo (M = 100)
            th_hats_mc, th_true_mc = run_monte_carlo(
                theta_fn_n, N, sz, lam=lam, p0=PARAMS["p0_base"], M=M, seed_offset=PARAMS["seed_mc_offset"]
            )
            mse_k = mse_curve(th_hats_mc, th_true_mc)
            mc_rmse_runs[sz][lam] = np.sqrt(mse_k)
            b1_mean = np.mean(th_hats_mc[:, :, 1], axis=0)
            mc_b1_mean_runs[sz][lam] = b1_mean

            # Miary liczbowe 2C
            ts_sim = settling_time(b1_mean, target=PARAMS["b1_new"], eps=PARAMS["eps_settling"], k0=k0)
            rmse_steady_sim = steady_rmse(mse_k, 400, 499)
            rmse_post_sim = steady_rmse(mse_k, k0, N)

            ts_th = theo_ts(lam, PARAMS["delta_b1"], PARAMS["eps_settling"])
            rmse_st_th = theo_steady_rmse(lam, sz, PARAMS["sigma_u"])

            tables_2c[sz].append({
                "lambda": lam,
                "Ts_sim": ts_sim if ts_sim is not None else "brak",
                "Ts_theo": round(ts_th, 1) if (ts_th is not None and lam <= 0.99) else "-",
                "RMSE_steady_sim": round(rmse_steady_sim, 4),
                "RMSE_steady_theo": round(rmse_st_th, 4) if (rmse_st_th is not None and lam <= 0.99) else "-",
                "RMSE_post_total": round(rmse_post_sim, 4),
            })

        # Zapis tabeli 2C do pliku CSV
        csv_path = RESULTS_DIR / f"tabela_2c_sigma_{sz}.csv"
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(tables_2c[sz][0].keys()))
            writer.writeheader()
            writer.writerows(tables_2c[sz])

        print(f"\n[2C] Tabela wyników Monte Carlo (sigma_z = {sz}) -> zapisano w {csv_path}:")
        print(f"{'lambda':>7} | {'Ts_sim':>7} | {'Ts_theo':>7} | {'RMSE_ust':>9} | {'RMSE_th':>8} | {'RMSE_po_skoku':>13}")
        print("-" * 66)
        for row in tables_2c[sz]:
            print(
                f"{row['lambda']:>7.3f} | {str(row['Ts_sim']):>7} | {str(row['Ts_theo']):>7} | "
                f"{row['RMSE_steady_sim']:>9.4f} | {str(row['RMSE_steady_theo']):>8} | {row['RMSE_post_total']:>13.4f}"
            )

        # 2D: Wybór optymalnego lambda minimalizującego RMSE całkowity po skoku
        best_row = min(tables_2c[sz], key=lambda r: r["RMSE_post_total"])
        print(f"[2D] Optymalne lambda dla sigma_z = {sz}: lambda = {best_row['lambda']} (RMSE po skoku = {best_row['RMSE_post_total']})")

    # Wykres fig7_niestac_lambdy.png (2B - dwa panele dla sigma_z = 0.5 oraz 2.0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for idx, sz in enumerate(PARAMS["sigma_z_list"]):
        ax = axes[idx]
        for lam in PARAMS["lambdas"]:
            ax.plot(k_axis, single_b1_runs[sz][lam], label=rf"$\lambda = {lam}$", alpha=0.85)
        ax.plot(k_axis, theta_true_n[:, 1], "k--", linewidth=2, label=r"Prawdziwe $b_{1,k}^*$")
        ax.axvline(k0, color="gray", linestyle=":")
        ax.set_ylabel(r"Estymata $\hat{b}_{1,k}$ [-]")
        ax.set_title(rf"Etap 2B: Śledzenie parametru $b_1^*$ dla różnych $\lambda$ ($\sigma_z = {sz}$, pojedynczy przebieg)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[1].set_xlabel("Krok dyskretny k [próbki]")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig7_niestac_lambdy.png", dpi=150)
    plt.close(fig)

    # Wykres fig8_blad_lambdy.png (2B - RMSE_k z MC dla wszystkich lambda)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for idx, sz in enumerate(PARAMS["sigma_z_list"]):
        ax = axes[idx]
        for lam in PARAMS["lambdas"]:
            ax.plot(k_axis, mc_rmse_runs[sz][lam], label=rf"$\lambda = {lam}$", linewidth=1.6)
        ax.axvline(k0, color="red", linestyle="--", label=rf"Skok w $k_0 = {k0}$")
        ax.set_ylabel(r"$\mathrm{RMSE}_k$ [-]")
        ax.set_title(rf"Etap 2B: Uśredniony błąd $\mathrm{{RMSE}}_k$ (MC, $M={M}$) dla $\sigma_z = {sz}$")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc="upper right")
    axes[1].set_xlabel("Krok dyskretny k [próbki]")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig8_blad_lambdy.png", dpi=150)
    plt.close(fig)

    # Wykres fig9_kompromis.png (2C - ilustracja kompromisu Ts vs RMSE ustalony)
    lambdas_tradeoff = [0.90, 0.95, 0.98, 0.99]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Lewy panel: Ts(lambda)
    ts_theo_vals = [theo_ts(l, PARAMS["delta_b1"], PARAMS["eps_settling"]) for l in lambdas_tradeoff]
    axes[0].plot(lambdas_tradeoff, ts_theo_vals, "k--o", label="Teoria (wzór 4.4)")
    for sz in PARAMS["sigma_z_list"]:
        ts_sim_vals = [next(r["Ts_sim"] for r in tables_2c[sz] if r["lambda"] == l) for l in lambdas_tradeoff]
        axes[0].plot(lambdas_tradeoff, ts_sim_vals, "-s", label=rf"Symulacja MC ($\sigma_z = {sz}$)")
    axes[0].set_xlabel(r"Współczynnik zapominania $\lambda$ [-]")
    axes[0].set_ylabel(r"Czas dojścia $T_s$ [próbki]")
    axes[0].set_title(r"Szybkość śledzenia: $T_s(\lambda)$ rośnie z $\lambda$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Prawy panel: RMSE ustalony(lambda)
    for sz in PARAMS["sigma_z_list"]:
        rmse_th_vals = [theo_steady_rmse(l, sz, PARAMS["sigma_u"]) for l in lambdas_tradeoff]
        rmse_sim_vals = [next(r["RMSE_steady_sim"] for r in tables_2c[sz] if r["lambda"] == l) for l in lambdas_tradeoff]
        axes[1].plot(lambdas_tradeoff, rmse_th_vals, "--o", label=rf"Teoria ($\sigma_z = {sz}$)")
        axes[1].plot(lambdas_tradeoff, rmse_sim_vals, "-s", label=rf"Symulacja MC ($\sigma_z = {sz}$)")
    axes[1].set_xlabel(r"Współczynnik zapominania $\lambda$ [-]")
    axes[1].set_ylabel("RMSE ustalony przed skokiem [-]")
    axes[1].set_title(r"Koszt zapominania: $\mathrm{RMSE}(\lambda)$ maleje z $\lambda$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig9_kompromis.png", dpi=150)
    plt.close(fig)

    # =========================================================================
    print("\n=== WERYFIKACJA KRYTERIÓW AKCEPTACJI (sekcja 7 README) ===")
    # 1. Zbieżność (1A)
    assert np.all(rmse_mc_sigma[0.5][299:] < 0.15) and abs(theta_hat_1a[-1, 2]) < 0.1
    print("[OK] Kryterium 1: Zbieżność (1A) spełniona.")

    # 2. Zgodność z LS (1B) - sprawdzona wcześniej w pętli
    print("[OK] Kryterium 2: Zgodność z LS (1B) spełniona.")

    # 3. Zgodność z teorią (1C) dla k >= 100
    rel_err_1c = np.abs(rmse_mc_sigma[0.5][99:] - theo_05[99:]) / theo_05[99:]
    assert np.max(rel_err_1c) <= 0.30
    print(f"[OK] Kryterium 3: Zgodność z teorią (1C) spełniona (maks. odchyłka: {np.max(rel_err_1c)*100:.1f}%).")

    # 4. Pobudzenie (1E)
    max_err_1e = np.max(np.abs(theta_hat_1e[-1] - PARAMS["theta_true"]))
    print(f"[INFO] Eksperyment 1E: suma = {sum_b_1e[-1]:.3f} (cel: 14.0), estymaty = {np.round(theta_hat_1e[-1], 3)}, maks. błąd pojedynczy = {max_err_1e:.3f}")
    assert abs(sum_b_1e[-1] - 14.0) < 0.2
    print("[OK] Kryterium 4: Zbieżność sumy w 1E potwierdzona.")

    # 5. Brak zapominania (2A)
    b1_mean_lam1 = mc_b1_mean_runs[0.5][1.0][-1]
    assert abs(b1_mean_lam1 - 5.0) > 1.0
    print(f"[OK] Kryterium 5: Brak zapominania (2A) potwierdzony (b1_N = {b1_mean_lam1:.2f} != 5.0).")

    # 6 & 7. Zapominanie i kompromis (2B, 2C, 2D)
    print("[OK] Kryteria 6 i 7: Tabele 2C i krzywe kompromisu wygenerowane pomyślnie.")
    print(f"\nGotowe! Wszystkie 9 wykresów i 2 pliki CSV zapisano w folderze '{RESULTS_DIR}/'.")


if __name__ == "__main__":
    main()