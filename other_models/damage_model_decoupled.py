import functools
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

from jax_fem.problem import Problem

jax.config.update("jax_enable_x64", True)


def smooth_plus(x, eps=1e-8):
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


class DamageModelCore:
    def _set_material_and_damage(self, E, nu, kappa_thresh, eta, c_d, beta_d, local_only=False):
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

        self.kappa_threshold = kappa_thresh
        self.eta = eta
        self.local_only = bool(local_only)
        self.gamma_d = 0.0 if self.local_only else 1.0
        self.c_d = 0.0 if self.local_only else float(c_d)
        self.beta_d = beta_d

    def damage_function(self, kappa):
        return jnp.exp(-self.eta * smooth_plus(kappa - self.kappa_threshold))

    def psi_neohookean(self, F):
        det_F = jnp.linalg.det(F)
        C = F @ F.T
        I1 = jnp.trace(C)
        log_J = jnp.log(det_F)
        return (self.mu / 2.0) * (I1 - 3.0) - self.mu * log_J + (self.lmbda / 2.0) * log_J ** 2

    def compute_PK1_base(self, F, _phi_grad, _phi, _kappa):
        det_F = jnp.linalg.det(F)
        log_J = jnp.log(det_F)
        F_inv_T = jnp.linalg.inv(F).T
        return self.mu * F + (self.lmbda * log_J - self.mu) * F_inv_T

    def compute_cauchy_stress(self, F, phi_grad, phi, kappa):
        P = self.compute_PK1_base(F, phi_grad, phi, kappa)
        det_F = jnp.linalg.det(F)
        return jnp.matmul(P, jnp.swapaxes(F, -1, -2)) / det_F

    def local_update_kappa(self, sed, phi_val, k_old):
        if self.local_only:
            return k_old + smooth_plus(sed - k_old)

        newton_tol = 1e-12
        max_iters = 100

        def phi_d(dlambda):
            kappa_trial = k_old + dlambda
            return (
                sed
                + self.beta_d * (phi_val - kappa_trial) / (self.eta * self.damage_function(kappa_trial))
                - kappa_trial
            )

        dphi_d = jax.grad(phi_d)
        dl0 = jnp.zeros((), dtype=jnp.result_type(sed, phi_val, k_old))
        r0 = phi_d(dl0)

        def cond_fn(state):
            dlambda, iters = state
            return (iters < max_iters) & (jnp.abs(phi_d(dlambda)) > newton_tol)

        def body_fn(state):
            dlambda, iters = state
            dlambda_new = smooth_plus(dlambda - phi_d(dlambda) / dphi_d(dlambda))
            return dlambda_new, iters + 1

        dlambda_final, _ = lax.while_loop(cond_fn, body_fn, (dl0, jnp.int32(0)))
        kappa_candidate = k_old + smooth_plus(dlambda_final)
        return lax.cond(r0 <= 0.0, lambda _: k_old, lambda _: kappa_candidate, operand=None)


class DisplacementProblemDecoupled(DamageModelCore, Problem):
    __hash__ = object.__hash__
    __eq__ = lambda _self, other: _self is other

    def custom_init(self, E, nu, kappa_thresh, eta, c_d, beta_d, local_only=False):
        self.fe_u = self.fes[0]
        self._set_material_and_damage(E, nu, kappa_thresh, eta, c_d, beta_d, local_only=local_only)
        self.internal_vars = [jnp.zeros((self.fe_u.num_cells, self.fe_u.num_quads))]

    def set_kappa_field(self, kappa_field):
        self.internal_vars = [kappa_field]

    def get_universal_kernel(self):
        def kernel(cell_sol_flat, _X, cell_shape_grads, _cell_JxW, cell_v_grads_JxW, kappa_elem):
            cell_sol_u = self.unflatten_fn_dof(cell_sol_flat)[0]
            cell_shape_grads_u = cell_shape_grads[:, :self.num_nodes_cumsum[1], :]
            cell_v_grads_u = cell_v_grads_JxW[:, :self.num_nodes_cumsum[1], :, :]

            u_grads = jnp.sum(cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :], axis=1)
            F = u_grads + jnp.eye(self.dim)

            zeros_grad = jnp.zeros((F.shape[0], self.dim))
            zeros_phi = jnp.zeros((F.shape[0],), dtype=F.dtype)
            P_int = jax.vmap(self.compute_PK1_base)(F, zeros_grad, zeros_phi, kappa_elem)
            res_u = jnp.sum(P_int[:, None, :, :] * cell_v_grads_u, axis=(0, -1))
            return jax.flatten_util.ravel_pytree([res_u])[0]

        return kernel

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_sed_field(self, u_sol):
        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)
        return jax.vmap(jax.vmap(self.psi_neohookean))(F)

    def compute_cauchy_quads(self, u_sol):
        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)
        zeros_grad = jnp.zeros((F.shape[0], F.shape[1], self.dim), dtype=F.dtype)
        zeros_phi = jnp.zeros((F.shape[0], F.shape[1]), dtype=F.dtype)
        kappa = self.internal_vars[0]
        return jax.vmap(jax.vmap(self.compute_cauchy_stress))(F, zeros_grad, zeros_phi, kappa)

    def compute_von_mises(self, u_sol):
        sigma = self.compute_cauchy_quads(u_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(self.dim)
        s_dev = sigma - hydrostatic
        von_mises = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))
        return jnp.mean(von_mises, axis=1)

    def compute_nodal_von_mises(self, u_sol):
        sigma = self.compute_cauchy_quads(u_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(self.dim)
        s_dev = sigma - hydrostatic
        von_mises_quad = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))
        return self.convert_quad_to_nodes(von_mises_quad)

    def convert_quad_to_nodes(self, quad_data):
        quad_data_np = np.array(quad_data)
        shape_vals = self.fe_u.shape_vals
        cells = self.fe_u.cells
        num_nodes_total = self.fe_u.num_total_nodes

        cell_nodal_values = np.sum(quad_data_np[:, :, None] * shape_vals[None, :, :], axis=1)
        flat_values = cell_nodal_values.flatten()
        flat_global_nodes = cells.flatten()

        nodal_sum = np.zeros(num_nodes_total)
        nodal_count = np.zeros(num_nodes_total, dtype=np.int32)
        np.add.at(nodal_sum, flat_global_nodes, flat_values)
        np.add.at(nodal_count, flat_global_nodes, 1)

        nodal_data = np.zeros(num_nodes_total)
        mask = nodal_count > 0
        nodal_data[mask] = nodal_sum[mask] / nodal_count[mask]
        return jnp.array(nodal_data)


class PhaseFieldProblemDecoupled(DamageModelCore, Problem):
    __hash__ = object.__hash__
    __eq__ = lambda _self, other: _self is other

    def custom_init(self, E, nu, kappa_thresh, eta, c_d, beta_d, local_only=False):
        self.fe_phi = self.fes[0]
        self._set_material_and_damage(E, nu, kappa_thresh, eta, c_d, beta_d, local_only=local_only)
        self.internal_vars = [jnp.zeros((self.fe_phi.num_cells, self.fe_phi.num_quads))]

    def set_kappa_field(self, kappa_field):
        self.internal_vars = [kappa_field]

    def get_universal_kernel(self):
        def kernel(cell_sol_flat, _X, cell_shape_grads, cell_JxW, cell_v_grads_JxW, kappa_elem):
            cell_sol_phi = self.unflatten_fn_dof(cell_sol_flat)[0]
            cell_shape_grads_phi = cell_shape_grads[:, :self.num_nodes_cumsum[1], :]
            cell_v_grads_phi = cell_v_grads_JxW[:, :self.num_nodes_cumsum[1], :, :]
            cell_JxW_phi = cell_JxW[0]

            shape_vals_phi = self.fe_phi.shape_vals
            phi_quad = jnp.sum(shape_vals_phi[:, :, None] * cell_sol_phi[None, :, :], axis=1)
            phi_vals = phi_quad[:, 0]
            phi_grads = jnp.sum(cell_sol_phi[None, :, :, None] * cell_shape_grads_phi[:, :, None, :], axis=1)

            grad_res = self.c_d * jnp.einsum("q1j,qn0j->n", phi_grads, cell_v_grads_phi)[:, None]
            phi_diff = phi_vals - self.gamma_d * kappa_elem
            penalty_term = phi_diff[:, None] * shape_vals_phi * cell_JxW_phi[:, None]
            penalty_res = self.beta_d * jnp.sum(penalty_term, axis=0)[:, None]

            return jax.flatten_util.ravel_pytree([grad_res + penalty_res])[0]

        return kernel

    @functools.partial(jax.jit, static_argnums=(0,))
    def local_update_kappa_from_sed(self, sed_field, phi_sol, kappa_n):
        phi_quad = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi_vals = jnp.squeeze(phi_quad, axis=-1)
        per_quad_update = jax.vmap(
            jax.vmap(self.local_update_kappa, in_axes=(0, 0, 0)),
            in_axes=(0, 0, 0),
        )
        return per_quad_update(sed_field, phi_vals, kappa_n)
