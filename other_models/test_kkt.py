import jax.numpy as jnp, jax

def smooth_plus(x, eps=1e-8):
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))

def smooth_plus_prime(x, eps=1e-8):
    return 0.5 * (1.0 + x / jnp.sqrt(x * x + eps * eps))

def _kappa_update_kkt_impl(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale):
    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    def Phi_d(dlambda):
        kappa_trial = k_old + dlambda
        fd = damage_function(kappa_trial)
        safe_fd = jnp.maximum(fd, 1e-150)
        penalty = jnp.where(phi_kappa_scale == 0.0, 0.0, phi_kappa_scale * beta_d * (phi - kappa_trial) / (eta * safe_fd))
        return sed + penalty - kappa_trial

    dPhi_d = jax.grad(Phi_d)
    dl0 = jnp.zeros((), dtype=jnp.result_type(sed, phi, k_old))
    R0 = Phi_d(dl0)
    active0 = R0 > 0.0

    def cond_fun(state):
        dl, it = state
        return (it < max_iters) & (jnp.abs(Phi_d(dl)) > newton_tol)

    def body_fun(state):
        dl, it = state
        step = Phi_d(dl) / dPhi_d(dl)
        return (dl - step, it + 1)

    dl_sol, _ = jax.lax.while_loop(cond_fun, body_fun, (dl0, jnp.int32(0)))
    k_new = jax.lax.cond(active0, lambda _: k_old + jnp.maximum(dl_sol, 0.0), lambda _: k_old, operand=None)
    return k_new

def _kappa_implicit_gradients(active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale):
    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    def active_grads():
        fd = damage_function(kappa_new)
        safe_fd = jnp.maximum(fd, 1e-150)
        inv_fd = 1.0 / safe_fd
        dsp = smooth_plus_prime(kappa_new - kappa_thresh, eps=eps_sp)
        
        dr_dsed = 1.0
        dr_dphi = jnp.where(phi_kappa_scale == 0.0, 0.0, phi_kappa_scale * beta_d * inv_fd / eta)

        dinv_fd_dk = inv_fd * eta * dsp
        dt_dk = jnp.where(phi_kappa_scale == 0.0, 0.0, (-1.0) * inv_fd / eta + (phi - kappa_new) * dinv_fd_dk / eta)
        dr_dk = jnp.where(phi_kappa_scale == 0.0, -1.0, phi_kappa_scale * beta_d * dt_dk - 1.0)

        dk_dsed = -dr_dsed / dr_dk
        dk_dphi = -dr_dphi / dr_dk
        # JAX requires returning 0.0 instead of int 0 when returning custom grads
        return dk_dsed, dk_dphi, jnp.zeros_like(dk_dsed)

    return jax.lax.cond(active, lambda _: active_grads(), lambda _: (jnp.zeros_like(sed), jnp.zeros_like(sed), jnp.ones_like(sed)), operand=None)

@jax.custom_jvp
def kappa_update_kkt(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp=1e-8, newton_tol=1e-12, max_iters=100, phi_kappa_scale=1.0):
    return _kappa_update_kkt_impl(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale)

@kappa_update_kkt.defjvp
def kappa_update_kkt_jvp(primals, tangents):
    sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale = primals
    sed_dot, phi_dot, k_old_dot, *_ = tangents

    kappa_new = _kappa_update_kkt_impl(sed, phi, k_old, kappa_thresh, eta, beta_d, eps_sp, newton_tol, max_iters, phi_kappa_scale)

    def damage_function(kappa):
        return jnp.exp(-eta * smooth_plus(kappa - kappa_thresh, eps=eps_sp))

    fd0 = damage_function(k_old)
    safe_fd0 = jnp.maximum(fd0, 1e-150)
    penalty0 = jnp.where(phi_kappa_scale == 0.0, 0.0, phi_kappa_scale * beta_d * (phi - k_old) / (eta * safe_fd0))
    active = (sed + penalty0 - k_old) > 0.0

    dk_dsed, dk_dphi, dk_dkold = _kappa_implicit_gradients(active, kappa_new, phi, sed, k_old, kappa_thresh, eta, beta_d, eps_sp, phi_kappa_scale)
    kappa_dot = dk_dsed * sed_dot + dk_dphi * phi_dot + dk_dkold * k_old_dot
    return kappa_new, kappa_dot

print(jax.jacobian(kappa_update_kkt, argnums=0)(jnp.array(500000.0), jnp.array(0.0), jnp.array(100.0), 0.1, 0.002, 1000.0, 1e-8, 1e-12, 100, 0.0))
