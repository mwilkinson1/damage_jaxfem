import jax
import jax.numpy as jnp
import jax.lax as lax


def smooth_plus(x, eps=1e-8):
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))

def smooth_plus_prime(x, eps=1e-8):
    # derivative of 0.5*(x + sqrt(x^2 + eps^2))
    return 0.5 * (1.0 + x / jnp.sqrt(x * x + eps * eps))


def _kappa_update_kkt_impl(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale):
    """Pure forward pass (no custom ad). Used by both primal and JVP.
    When phi_kappa_scale=0 (local_only), Phi_d reduces to sed - kappa_trial."""
    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    def Phi_d(dlambda):
        kappa_trial = k_old + dlambda
        fd = damage_function(kappa_trial)
        safe_fd = jnp.maximum(fd, 1e-13)
        penalty = jnp.where(
            phi_kappa_scale == 0.0,
            jnp.zeros_like(sed),
            phi_kappa_scale * beta_d * (phi - kappa_trial) / (eta * safe_fd)
        )
        return sed + penalty - kappa_trial

    dPhi_d = jax.grad(Phi_d)

    dl0 = jnp.zeros((), dtype=jnp.result_type(sed, phi, k_old))
    R0 = Phi_d(dl0)
    active0 = R0 > 0.0

    def _run_safeguarded_root_and_return():
        max_bracket_iters = max_iters
        lower = dl0
        r_lower = R0
        init_upper = jnp.maximum(jnp.maximum(sed - k_old, 0.0), 1.0)
        r_init_upper = Phi_d(init_upper)

        def bracket_cond(state):
            _, _, r_upper, it = state
            return (it < max_bracket_iters) & (r_upper > 0.0)

        def bracket_body(state):
            _, upper, _, it = state
            upper_new = 2.0 * upper
            r_upper_new = Phi_d(upper_new)
            return lower, upper_new, r_upper_new, it + 1

        lower, upper, r_upper, _ = lax.while_loop(
            bracket_cond,
            bracket_body,
            (lower, init_upper, r_init_upper, jnp.int32(0)),
        )

        has_valid_bracket = r_upper <= 0.0

        def solve_in_bracket(_):
            r_lo0 = r_lower
            r_hi0 = r_upper
            dl_sol0 = 0.5 * (lower + upper)

            def solve_cond(state):
                _, _, _, dl_sol, it = state
                return (it < max_iters) & (jnp.abs(Phi_d(dl_sol)) > newton_tol)

            def solve_body(state):
                lo, hi, r_lo, dl_sol, it = state
                r_sol = Phi_d(dl_sol)
                deriv = dPhi_d(dl_sol)
                use_newton = jnp.abs(deriv) > 1e-14
                newton_dl = dl_sol - r_sol / jnp.where(use_newton, deriv, 1.0)
                bisect_dl = 0.5 * (lo + hi)
                candidate = jnp.where(
                    use_newton & (newton_dl > lo) & (newton_dl < hi),
                    newton_dl,
                    bisect_dl,
                )
                r_candidate = Phi_d(candidate)
                lo_new = jnp.where(r_candidate > 0.0, candidate, lo)
                hi_new = jnp.where(r_candidate > 0.0, hi, candidate)
                r_lo_new = jnp.where(r_candidate > 0.0, r_candidate, r_lo)
                return lo_new, hi_new, r_lo_new, candidate, it + 1

            _, _, _, dl_sol, _ = lax.while_loop(
                solve_cond,
                solve_body,
                (lower, upper, r_lo0, dl_sol0, jnp.int32(0)),
            )
            return k_old + jnp.maximum(dl_sol, 0.0)

        def fallback_inactive(_):
            return k_old

        return lax.cond(has_valid_bracket, solve_in_bracket, fallback_inactive, operand=None)

    def _inactive_return_k_old():
        return k_old

    return lax.cond(active0, _run_safeguarded_root_and_return, _inactive_return_k_old)


def _kappa_implicit_gradients(active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale):
    """Compute dk/dsed, dk/dphi, dk/dkold for JVP and VJP."""
    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    def active_grads():
        fd = damage_function(kappa_new)
        safe_fd = jnp.maximum(fd, 1e-15)
        inv_fd = 1.0 / safe_fd
        dsp = smooth_plus_prime(kappa_new - kappa_thresh, eps=eps_sp)
        dfd_dk = fd * (-eta) * dsp

        dr_dsed = jnp.ones_like(sed)
        
        dr_dphi = jnp.where(
            phi_kappa_scale == 0.0,
            jnp.zeros_like(sed),
            phi_kappa_scale * beta_d * inv_fd / eta
        )

        dinv_fd_dk = -inv_fd * inv_fd * dfd_dk
        
        dt_dk = jnp.where(
            phi_kappa_scale == 0.0,
            jnp.zeros_like(sed),
            (-1.0) * inv_fd / eta + (phi - kappa_new) * dinv_fd_dk / eta
        )
        
        dr_dk = jnp.where(
            phi_kappa_scale == 0.0,
            -jnp.ones_like(sed),
            phi_kappa_scale * beta_d * dt_dk - 1.0
        )

        dr_dk = jnp.asarray(dr_dk)
        dr_dphi = jnp.asarray(dr_dphi)
        dk_dsed = -dr_dsed / dr_dk
        dk_dphi = -dr_dphi / dr_dk
        dk_dkold = jnp.zeros_like(sed)
        return dk_dsed, dk_dphi, dk_dkold

    def inactive_grads():
        zeros = jnp.zeros_like(sed)
        ones = jnp.ones_like(sed)
        return zeros, zeros, ones

    return lax.cond(active, lambda _: active_grads(), lambda _: inactive_grads(), operand=None)


@jax.custom_jvp
def kappa_update_kkt(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp=1e-8, newton_tol=1e-12, max_iters=100, phi_kappa_scale=1.0):
    """
    Forward: active-set + local Newton solve.
    When phi_kappa_scale=0 (e.g. local_only), the phi-kappa term is disabled and
    the residual reduces to sed - kappa_trial.
    JVP: implicit differentiation in active regime; identity in inactive.
    VJP is derived by transposition from JVP (works with jacfwd and jacrev).
    """
    return _kappa_update_kkt_impl(
        sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale
    )


@kappa_update_kkt.defjvp
def kappa_update_kkt_jvp(primals, tangents):
    sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale = primals
    sed_dot, phi_dot, k_old_dot, *_ = tangents

    kappa_new = _kappa_update_kkt_impl(
        sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale
    )

    active = kappa_new > (k_old + 1e-14)

    dk_dsed, dk_dphi, dk_dkold = _kappa_implicit_gradients(
        active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale
    )
    kappa_dot = dk_dsed * sed_dot + dk_dphi * phi_dot + dk_dkold * k_old_dot

    return kappa_new, kappa_dot


def kappa_update_kkt_with_gradients(
    sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp=1e-8, newton_tol=1e-12, max_iters=100, phi_kappa_scale=1.0
):
    """
    Returns (kappa, (dk_dsed, dk_dphi, dk_dkold)) so derivatives can be used
    without differentiating through kappa (e.g. for consistent tangents).
    """
    kappa_new = _kappa_update_kkt_impl(
        sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale
    )

    active = kappa_new > (k_old + 1e-14)

    dk_dsed, dk_dphi, dk_dkold = _kappa_implicit_gradients(
        active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale
    )
    return kappa_new, (dk_dsed, dk_dphi, dk_dkold)


def compute_dr_dk_for_diagnostics(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp=1e-8, phi_kappa_scale=1.0):
    """
    Diagnostic: compute dr_dk (derivative of Phi_d w.r.t. kappa at kappa_new) for each quad.
    Returns (kappa_new, dr_dk) with same shape as sed. Used to check for near-zero dr_dk (Hypothesis B).
    """
    kappa_new = _kappa_update_kkt_impl(
        sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, 1e-12, 100, phi_kappa_scale
    )
    active = kappa_new > (k_old + 1e-14)

    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    def dr_dk_active():
        fd = damage_function(kappa_new)
        safe_fd = jnp.maximum(fd, 1e-15)
        inv_fd = 1.0 / safe_fd
        dsp = smooth_plus_prime(kappa_new - kappa_thresh, eps=eps_sp)
        dfd_dk = fd * (-eta) * dsp
        dinv_fd_dk = -inv_fd * inv_fd * dfd_dk
        dt_dk = jnp.where(
            phi_kappa_scale == 0.0,
            jnp.zeros_like(sed),
            (-1.0) * inv_fd / eta + (phi - kappa_new) * dinv_fd_dk / eta
        )
        return jnp.where(
            phi_kappa_scale == 0.0,
            -jnp.ones_like(sed),
            phi_kappa_scale * beta_d * dt_dk - 1.0
        )

    dr_dk = jnp.where(active, dr_dk_active(), jnp.ones_like(sed))
    return kappa_new, dr_dk

