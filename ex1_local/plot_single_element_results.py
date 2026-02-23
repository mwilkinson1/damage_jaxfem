"""
Generate figures from single_element_results (local-only damage) using arc-length
continuation results: Cauchy stress sigma_11, kappa, and damage d vs local stretch.

Run single_element_arclength.py first to produce arclength_*.csv results.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent / "single_element_results"
DOMAIN_SIZE = 1.0

# Material parameters (from single_element.py)
E = 42.0  # MPa
NU = 0.45
MU = E / (2.0 * (1.0 + NU))

# Parameter sweep values (must match single_element.py)
ETA_VALS = [1.0, 2.0, 5.0, 20.0, 100.0]  # MPa^-1
KAPPA_VALS = [0.0, 0.2, 0.6, 1.0, 2.0]  # MPa
KAPPA_THRESH_ETA = 0.2  # MPa (for eta sweep)
ETA_FIXED = 2.0  # MPa^-1 (for kappa sweep)

# Line styles: first two solid (blue, red), rest dashed (black, blue, red)
ETA_STYLES = [
    ("#1f77b4", "-", 1.5),   # blue solid
    ("#d62728", "-", 1.5),   # red solid
    ("#000000", "--", 1.0),  # black dashed
    ("#1f77b4", "--", 1.0),  # blue dashed
    ("#d62728", "--", 1.0),  # red dashed
]
KAPPA_STYLES = ETA_STYLES


def undamaged_sigma_11(lambda_vals):
    """Neo-Hookean Cauchy stress for uniaxial extension (undamaged)."""
    return MU * (lambda_vals**2 - 1.0 / lambda_vals)


def load_csv(filepath):
    """Load CSV results; return dict with arrays."""
    with open(filepath) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return {
        "increment": np.array([float(r["increment"]) for r in rows]),
        "prescribed_displacement": np.array([float(r["prescribed_displacement"]) for r in rows]),
        "phi_mean": np.array([float(r["phi_mean"]) for r in rows]),
        "kappa_mean": np.array([float(r["kappa_mean"]) for r in rows]),
        "damage_mean": np.array([float(r["damage_mean"]) for r in rows]),
        "von_mises_mean": np.array([float(r["von_mises_mean"]) for r in rows]),
    }


def lambda_from_displacement(displacement):
    """Local stretch from prescribed displacement (uniaxial x)."""
    return 1.0 + displacement / DOMAIN_SIZE


def load_eta_sweep_data():
    """Load arc-length data for eta_d sweep (fixed kappa_d = 0.2)."""
    data = {}
    for eta in ETA_VALS:
        path = RESULTS_DIR / f"arclength_eta_{eta:.1f}_kappa_{KAPPA_THRESH_ETA:.1f}.csv"
        if path.exists():
            data[eta] = load_csv(path)
    return data


def load_kappa_sweep_data():
    """Load arc-length data for kappa_d sweep (fixed eta_d = 2.0)."""
    data = {}
    for kappa in KAPPA_VALS:
        path = RESULTS_DIR / f"arclength_eta_{ETA_FIXED:.1f}_kappa_{kappa:.1f}.csv"
        if path.exists():
            data[kappa] = load_csv(path)
    return data


def plot_sigma_11_figure(eta_data, kappa_data, out_path):
    """Create Fig. 7: sigma_11 vs lambda, two subplots (local-only)."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))

    lambda_range = np.linspace(1.0, 1.5, 100)
    undamaged = undamaged_sigma_11(lambda_range)

    for ax in (ax_left, ax_right):
        ax.plot(lambda_range, undamaged, "k--", lw=2.5, label="undamaged")
        ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel(r"local stretch $\lambda$ [-]")
        ax.set_ylabel(r"Cauchy stress $\sigma_{11}$ [MPa]")
        ax.set_xlim(1.0, 1.5)
        ax.set_ylim(0, 18)
        ax.grid(True, alpha=0.3)

    for idx, (eta, d) in enumerate(eta_data.items()):
        if d is None:
            continue
        lam = lambda_from_displacement(d["prescribed_displacement"])
        sigma = d["von_mises_mean"]
        color, ls, lw = ETA_STYLES[idx]
        ax_left.plot(lam, sigma, color=color, linestyle=ls, lw=lw,
                     label=rf"$\eta_d = {eta:.1f}$ MPa$^{{-1}}$")

    for idx, (kappa, d) in enumerate(kappa_data.items()):
        if d is None:
            continue
        lam = lambda_from_displacement(d["prescribed_displacement"])
        sigma = d["von_mises_mean"]
        color, ls, lw = KAPPA_STYLES[idx]
        ax_right.plot(lam, sigma, color=color, linestyle=ls, lw=lw,
                      label=rf"$\kappa_d = {kappa:.1f}$ MPa")

    ax_left.set_title(r"Varying $\eta_d$, fixed $\kappa_d = 0.2$ MPa (arc-length)")
    ax_right.set_title(r"Varying $\kappa_d$, fixed $\eta_d = 2.0$ MPa$^{-1}$ (arc-length)")
    ax_left.legend(loc="upper right", fontsize=8)
    ax_right.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kappa_d_figure(eta_data, kappa_data, out_path):
    """Create Fig. 8: kappa and d vs lambda, 2x2 grid (local-only)."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax_kappa_eta, ax_kappa_kappa = axes[0]
    ax_d_eta, ax_d_kappa = axes[1]

    for idx, (eta, d) in enumerate(eta_data.items()):
        if d is None:
            continue
        lam = lambda_from_displacement(d["prescribed_displacement"])
        kappa = d["kappa_mean"]
        damage = d["damage_mean"]
        color, ls, lw = ETA_STYLES[idx]
        label = rf"$\eta_d = {eta:.1f}$ MPa$^{{-1}}$"
        ax_kappa_eta.plot(lam, kappa, color=color, linestyle=ls, lw=lw, label=label)
        ax_d_eta.plot(lam, damage, color=color, linestyle=ls, lw=lw, label=label)

    for idx, (kappa, d) in enumerate(kappa_data.items()):
        if d is None:
            continue
        lam = lambda_from_displacement(d["prescribed_displacement"])
        kappa_val = d["kappa_mean"]
        damage = d["damage_mean"]
        color, ls, lw = KAPPA_STYLES[idx]
        label = rf"$\kappa_d = {kappa:.1f}$ MPa"
        ax_kappa_kappa.plot(lam, kappa_val, color=color, linestyle=ls, lw=lw, label=label)
        ax_d_kappa.plot(lam, damage, color=color, linestyle=ls, lw=lw, label=label)

    for ax in axes.flat:
        ax.set_xlabel(r"local stretch $\lambda$ [-]")
        ax.set_xlim(1.0, 1.5)
        ax.grid(True, alpha=0.3)

    ax_kappa_eta.set_ylabel(r"local damage variable $\kappa$ [MPa]")
    ax_kappa_eta.set_ylim(0, 1.2)
    ax_kappa_eta.legend(loc="upper left", fontsize=8)
    ax_kappa_eta.set_title(rf"Varying $\eta_d$, fixed $\kappa_d = {KAPPA_THRESH_ETA}$ MPa (arc-length)")

    ax_kappa_kappa.set_ylabel(r"local damage variable $\kappa$ [MPa]")
    ax_kappa_kappa.set_ylim(0, 2.4)
    ax_kappa_kappa.legend(loc="upper left", fontsize=8)
    ax_kappa_kappa.set_title(rf"Varying $\kappa_d$, fixed $\eta_d = {ETA_FIXED}$ MPa$^{{-1}}$ (arc-length)")

    ax_d_eta.set_ylabel(r"damage $d = 1 - f_d(\kappa)$ [-]")
    ax_d_eta.set_ylim(0, 1.0)
    ax_d_eta.legend(loc="upper left", fontsize=8)

    ax_d_kappa.set_ylabel(r"damage $d = 1 - f_d(\kappa)$ [-]")
    ax_d_kappa.set_ylim(0, 1.0)
    ax_d_kappa.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    results_dir = RESULTS_DIR
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    eta_data = load_eta_sweep_data()
    kappa_data = load_kappa_sweep_data()

    if not eta_data or not kappa_data:
        print("Missing arc-length CSV files. Run single_element_arclength.py first.")
        return

    out_sigma = results_dir / "fig_sigma11_vs_lambda.png"
    out_kappa_d = results_dir / "fig_kappa_d_vs_lambda.png"

    plot_sigma_11_figure(eta_data, kappa_data, out_sigma)
    print(f"Saved: {out_sigma}")

    plot_kappa_d_figure(eta_data, kappa_data, out_kappa_d)
    print(f"Saved: {out_kappa_d}")


if __name__ == "__main__":
    main()
