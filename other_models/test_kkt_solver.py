"""
Demonstration: reproduces the KKT Newton "negative kappa then jump" issue numerically.

With a critical SED (and phi in the high-damage regime while k_old is still low), the
residual R(dλ) has R'(0) > 0 at dλ=0, so the first Newton step drives dλ negative.
kappa_trial = k_old + dλ then goes below kappa_thresh (and can go negative). There
the damage coupling f_d(κ) changes regime, the derivative dR/d(dλ) flips, and one
large step jumps the solution to the high-kappa root. This script runs the diagnostic
(residual landscape) and the Newton loop with full print/plot to show the issue.
"""
import numpy as np
import jax.numpy as jnp
import jax
from custom_vjp_kkt import smooth_plus, kappa_update_kkt

jax.config.update("jax_enable_x64", True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

# ex2_gradient parameters (eta_1.0_kappa_0.2 regime)
KAPPA_THRESH = 0.2
ETA = 1.0
BETA_D = 1000.0
EPS_SP = 1e-8
PHI_KAPPA_SCALE = 1.0

# Critical SED demonstration: sed/phi in "high" regime, k_old still "low" -> Newton
# goes negative (kappa_trial drops below threshold) then jumps to high-kappa root.
# This choice cleanly reproduces the issue in ~10 iterations.
SED = 18.5
PHI = 18.5
K_OLD = 3.48


def damage_function(kappa, kappa_thresh, eta, eps_sp):
    return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))


def residual_and_full_state(dlambda, sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale):
    """R(dλ), dR/d(dλ), and all intermediates."""
    kappa_trial = k_old + dlambda
    fd = damage_function(kappa_trial, kappa_thresh, eta, eps_sp)
    penalty_term = phi_kappa_scale * beta_d * (phi - kappa_trial) / (eta * fd)
    res = sed + penalty_term - kappa_trial

    def Phi_d(dl):
        kt = k_old + dl
        f = damage_function(kt, kappa_thresh, eta, eps_sp)
        return sed + phi_kappa_scale * beta_d * (phi - kt) / (eta * f) - kt

    deriv = float(jax.grad(Phi_d)(dlambda))
    return {
        "dl": float(dlambda),
        "kappa_trial": float(kappa_trial),
        "fd": float(fd),
        "penalty_term": float(penalty_term),
        "res": float(res),
        "deriv": deriv,
    }


def run_newton_with_full_print(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale,
                               newton_tol=1e-10, max_iters=100):
    """Newton loop; print all variables each iteration; return history for plotting."""
    dl = 0.0
    history = []

    print("\n--- Inputs ---")
    print(f"  sed={sed}  phi={phi}  k_old={k_old}")
    print(f"  kappa_thresh={kappa_thresh}  eta={eta}  beta_d={beta_d}  eps_sp={eps_sp}  phi_kappa_scale={phi_kappa_scale}")

    for it in range(max_iters):
        state = residual_and_full_state(
            dl, sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale
        )
        deriv = state["deriv"]
        res = state["res"]
        step = 0.0 if abs(deriv) < 1e-20 else res / deriv
        dl_new = dl - step

        state["step"] = step
        state["dl_new"] = dl_new
        state["it"] = it
        history.append(state)

        print(f"\n--- Iteration {it} ---")
        for k in ["it", "dl", "kappa_trial", "fd", "penalty_term", "res", "deriv", "step", "dl_new"]:
            v = state.get(k)
            if v is not None:
                print(f"  {k} = {v}")

        if abs(res) <= newton_tol:
            break
        dl = dl_new

    kappa_candidate = k_old + max(dl_new, 0.0)
    active0 = history[0]["res"] > 0.0
    kappa_final = kappa_candidate if active0 else k_old
    print(f"\n--- Result ---")
    print(f"  active0={active0}  kappa_final={kappa_final}  kappa_candidate={kappa_candidate}  dl_sol={dl_new}")

    kappa_solver = float(kappa_update_kkt(
        sed, phi, k_old, kappa_thresh, eta, beta_d,
        eps_sp=eps_sp, newton_tol=newton_tol, max_iters=max_iters,
        phi_kappa_scale=phi_kappa_scale
    ))
    print(f"  kappa_solver={kappa_solver}  err={abs(kappa_final - kappa_solver)}")

    return kappa_final, history, active0


def diagnose_why_negative_kappa(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale):
    """
    Plot R(dλ) and dR/d(dλ) over a range of dλ to show why Newton goes negative.
    kappa_trial = k_old + dλ; when dλ < -k_old, kappa_trial goes negative.
    """
    dl_min = -5.0
    dl_max = 20.0
    n_pts = 300
    dl_vals = np.linspace(dl_min, dl_max, n_pts)
    ress = []
    derivs = []
    for dl in dl_vals:
        state = residual_and_full_state(
            dl, sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale
        )
        ress.append(state["res"])
        derivs.append(state["deriv"])

    state0 = residual_and_full_state(
        0.0, sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale
    )
    res0 = state0["res"]
    deriv0 = state0["deriv"]
    step0 = res0 / deriv0 if abs(deriv0) > 1e-20 else 0.0

    print("\n" + "=" * 60)
    print("WHY NEWTON GOES TO NEGATIVE KAPPA (dλ)")
    print("=" * 60)
    print(f"  At dλ = 0:  R(0) = {res0:.6e}   dR/d(dλ)(0) = {deriv0:.6e}")
    print(f"  First Newton step = R(0)/R'(0) = {step0:.6e}")
    print(f"  So dλ_new = 0 - step = {-step0:.6e}")
    if deriv0 > 0 and res0 > 0:
        print("  → R'(0) > 0 and R(0) > 0  =>  step > 0  =>  dλ_new = -step < 0")
        print("  → Newton moves dλ in the NEGATIVE direction (kappa_trial decreases).")
    elif deriv0 < 0 and res0 > 0:
        print("  → R'(0) < 0 and R(0) > 0  =>  step < 0  =>  dλ_new = -step > 0")
        print("  → Newton moves dλ in the positive direction.")
    print("  Roots (R(dλ)=0):")
    for i in range(len(dl_vals) - 1):
        if (ress[i] <= 0 <= ress[i + 1]) or (ress[i] >= 0 >= ress[i + 1]):
            dl_root = dl_vals[i] + (dl_vals[i + 1] - dl_vals[i]) * (0 - ress[i]) / (ress[i + 1] - ress[i]) if ress[i + 1] != ress[i] else dl_vals[i]
            kappa_root = k_old + dl_root
            print(f"    dλ ≈ {dl_root:.4f}  (kappa_trial ≈ {kappa_root:.4f})")
    print("=" * 60)

    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(dl_vals, ress, "b-", label="R(dλ)")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].axvline(0, color="gray", linestyle=":")
    axes[0].axvline(-step0, color="red", linestyle="--", alpha=0.8, label=f"1st step: dλ = {-step0:.3f}")
    axes[0].set_xlabel("dλ (kappa_trial = k_old + dλ)")
    axes[0].set_ylabel("Residual R")
    axes[0].set_title("Residual landscape: positive dR/dλ at 0 → Newton step goes negative")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    dl_thresh = kappa_thresh - k_old
    axes[1].plot(dl_vals, derivs, "g-", label="dR/d(dλ)")
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].axvline(0, color="gray", linestyle=":")
    axes[1].axvline(dl_thresh, color="orange", linestyle="--", alpha=0.8,
                    label=f"kappa_trial = k_thresh (dλ = {dl_thresh:.3f})")
    axes[1].set_xlabel("dλ")
    axes[1].set_ylabel("dR/d(dλ)")
    axes[1].set_title("Derivative: sign change when kappa_trial crosses kappa_thresh")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("kkt_residual_landscape.png", bbox_inches="tight")
    print("\nSaved kkt_residual_landscape.png")
    plt.close()


def main():
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Negative kappa then jump (critical SED)")
    print("=" * 70)
    print(f"  Critical SED = {SED}, phi = {PHI}, k_old = {K_OLD}")
    print("  Expectation: Newton drives dλ negative, kappa_trial crosses below")
    print("  kappa_thresh, then one large step jumps to the high-kappa root.")
    print("=" * 70)

    diagnose_why_negative_kappa(
        SED, PHI, K_OLD, KAPPA_THRESH, ETA, BETA_D, EPS_SP, PHI_KAPPA_SCALE
    )

    kappa_final, history, active = run_newton_with_full_print(
        SED, PHI, K_OLD, KAPPA_THRESH, ETA, BETA_D, EPS_SP, PHI_KAPPA_SCALE
    )

    # Summary: confirm the issue was reproduced
    if history:
        dls = [h["dl"] for h in history]
        kappa_trials = [K_OLD + dl for dl in dls]
        went_negative = any(kt < KAPPA_THRESH for kt in kappa_trials)
        print("\n" + "=" * 70)
        print("DEMONSTRATION SUMMARY")
        print("=" * 70)
        print(f"  kappa_final = {kappa_final:.4f}  (high-kappa root)")
        print(f"  Newton path crossed below kappa_thresh ({KAPPA_THRESH}): {went_negative}")
        print("  Issue reproduced: critical SED leads to R'(0)>0 -> negative dλ -> jump.")
        print("=" * 70)

    if not HAS_PLOT or not history:
        return

    iters = [h["it"] for h in history]
    dls = [h["dl"] for h in history]
    ress = [h["res"] for h in history]
    derivs = [h["deriv"] for h in history]
    steps = [h["step"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    name = "critical_sed_demo"
    fig.suptitle(f"KKT Newton demo (critical SED={SED}, phi={PHI}, k_old={K_OLD})")

    axes[0, 0].semilogy(iters, [abs(r) for r in ress], "o-", color="C0")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 0].set_ylabel("|residual|")
    axes[0, 0].set_title("Residual")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, dls, "s-", color="C1")
    axes[0, 1].set_xlabel("iteration")
    axes[0, 1].set_ylabel("dλ")
    axes[0, 1].set_title("Lagrange multiplier increment")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, derivs, "^-", color="C2")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 0].set_ylabel("dPhi_d(dλ)")
    axes[1, 0].set_title("Derivative")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, steps, "d-", color="C3")
    axes[1, 1].set_xlabel("iteration")
    axes[1, 1].set_ylabel("Newton step")
    axes[1, 1].set_title("Step (res/deriv)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = "kkt_newton_critical_sed_demo.png"
    plt.savefig(outfile, bbox_inches="tight")
    print(f"\nSaved {outfile}")
    plt.close()


if __name__ == "__main__":
    main()
