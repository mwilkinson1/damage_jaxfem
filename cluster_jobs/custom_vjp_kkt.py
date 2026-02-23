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
        return sed + phi_kappa_scale * beta_d * (phi - kappa_trial) / (eta * fd) - kappa_trial

    dPhi_d = jax.grad(Phi_d)

    dl0 = jnp.zeros((), dtype=jnp.result_type(sed, phi, k_old))
    R0 = Phi_d(dl0)
    active0 = R0 > 0.0

    def cond_fun(state):
        dl, it = state
        return (it < max_iters) & (jnp.abs(Phi_d(dl)) > newton_tol)

    def body_fun(state):
        dl, it = state
        res = Phi_d(dl)
        deriv = dPhi_d(dl)
        step = res / deriv
        dl_new = dl - step
        return (dl_new, it + 1)

    def _run_newton_and_return():
        dl_sol, _ = lax.while_loop(cond_fun, body_fun, (dl0, jnp.int32(0)))
        return k_old + jnp.maximum(dl_sol, 0.0)

    def _inactive_return_k_old():
        return k_old

    return lax.cond(active0, _run_newton_and_return, _inactive_return_k_old)


def _kappa_implicit_gradients(active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale):
    """Compute dk/dsed, dk/dphi, dk/dkold for JVP and VJP."""
    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    def active_grads():
        fd = damage_function(kappa_new)
        inv_fd = 1.0 / fd
        dsp = smooth_plus_prime(kappa_new - kappa_thresh, eps=eps_sp)
        dfd_dk = fd * (-eta) * dsp

        dr_dsed = 1.0
        dr_dphi = phi_kappa_scale * beta_d * inv_fd / eta

        dinv_fd_dk = -inv_fd * inv_fd * dfd_dk
        dt_dk = (-1.0) * inv_fd / eta + (phi - kappa_new) * dinv_fd_dk / eta
        dr_dk = phi_kappa_scale * beta_d * dt_dk - 1.0

        dk_dsed = -dr_dsed / dr_dk
        dk_dphi = -dr_dphi / dr_dk
        dk_dkold = 0.0
        return dk_dsed, dk_dphi, dk_dkold

    def inactive_grads():
        return 0.0, 0.0, 1.0

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

    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    fd0 = damage_function(k_old)
    R0 = sed + phi_kappa_scale * beta_d * (phi - k_old) / (eta * fd0) - k_old
    active = R0 > 0.0

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

    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    fd0 = damage_function(k_old)
    R0 = sed + phi_kappa_scale * beta_d * (phi - k_old) / (eta * fd0) - k_old
    active = R0 > 0.0

    dk_dsed, dk_dphi, dk_dkold = _kappa_implicit_gradients(
        active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale
    )
    return kappa_new, (dk_dsed, dk_dphi, dk_dkold)

