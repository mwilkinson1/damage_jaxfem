"""
Staggered damage: same as damage_model.py except the kernel uses the internal
variable kappa (previous step) instead of kappa_tmp from local_update_kappa.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax

from jax_fem.problem import Problem

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Safe positive/negative parts for autodiff (avoids kinks from max(x,0)/min(x,0) at 0)
def smooth_plus(x, eps=1e-8):
    return 0.5 * (x + jnp.sqrt(x*x + eps*eps))


class CoupledDamageProblemStaggered(Problem):

    __hash__ = object.__hash__
    __eq__ = lambda _self, other: _self is other

    def custom_init(self, E, nu, kappa_thresh, eta, c_d, beta_d, local_only=False):
        """Initialize material and damage parameters.

        local_only: 7th element of additional_info. If True, c_d=0 and gamma_d=0 so
        the penalty is (beta_d/2)*phi^2 → weak form gives phi=0; kappa is driven
        only by strain energy density (history).
        """
        self.fe_u = self.fes[0]
        self.fe_phi = self.fes[1]

        # Material properties
        self.E, self.nu = E, nu
        self.mu = E / (2 * (1 + nu))
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

        # Damage parameters
        self.kappa_threshold = kappa_thresh
        self.eta = eta
        self.local_only = bool(local_only)
        self.gamma_d = 0.0 if self.local_only else 1.0
        self.c_d = 0.0 if self.local_only else float(c_d)
        self.beta_d = beta_d
        self.eps_reg = 1e-10

        # Internal variables
        num_cells = self.fe_u.num_cells
        num_quads = self.fe_u.num_quads
        self.internal_vars = [jnp.zeros((num_cells, num_quads))]

    def _squeeze_phi_grad(self, phi_grad):
        """Ensure phi_grad has shape (..., dim)."""
        if phi_grad.ndim >= 2 and phi_grad.shape[-2] == 1:
            return jnp.squeeze(phi_grad, axis=-2)
        return phi_grad

    def _compute_C_inv(self, F):
        """Right Cauchy-Green inverse: C^{-1} with C = F^T F."""
        C = jnp.matmul(jnp.swapaxes(F, -1, -2), F)
        return jnp.linalg.inv(C)

    def damage_function(self, kappa):
        return jnp.exp(-self.eta * smooth_plus(kappa - self.kappa_threshold))

    def psi_neohookean(self, F):
        """Neo-Hookean strain energy."""
        det_F = jnp.linalg.det(F)
        C = F @ F.T
        I1 = jnp.trace(C)
        log_J = jnp.log(det_F)
        psi_val = (self.mu / 2.0) * (I1 - 3.0) - self.mu * log_J + (self.lmbda / 2.0) * log_J ** 2
        return psi_val

    def compute_PK1_base(self, F, phi_grad, phi, kappa):
        det_F = jnp.linalg.det(F)
        log_J = jnp.log(det_F)
        F_inv_T = jnp.linalg.inv(F).T
        PK1 = self.mu * F + (self.lmbda * log_J - self.mu) * F_inv_T
        return self.damage_function(kappa) * PK1

    def compute_cauchy_stress(self, F, phi_grad, phi, kappa):
        P = self.compute_PK1_base(F, phi_grad, phi, kappa)
        det_F = jnp.linalg.det(F)
        sigma = jnp.matmul(P, jnp.swapaxes(F, -1, -2)) / det_F
        return sigma

    def local_update_kappa(self, sed, phi_val, k_old):
        if self.local_only:
            return k_old + smooth_plus(sed - k_old)

        newton_tol = 1e-12
        max_iters = 100

        def Phi_d(dlambda):
            kappa_trial = k_old + dlambda
            return sed + self.beta_d * (phi_val - kappa_trial) / (self.eta * self.damage_function(kappa_trial)) - kappa_trial

        DPhi_d = jax.grad(Phi_d)

        dl0 = jnp.zeros((), dtype=jnp.result_type(sed, phi_val, k_old))
        R0 = Phi_d(dl0)

        def cond_fun(state):
            dl, iters = state
            residual = jnp.abs(Phi_d(dl))
            return (iters < max_iters) & (residual > newton_tol)

        def body_fun(state):
            dl, iters = state
            dl_new = smooth_plus(dl - Phi_d(dl) / DPhi_d(dl))
            return (dl_new, iters + 1)

        init_iters = jnp.int32(0)
        dl_final, _ = lax.while_loop(cond_fun, body_fun, (dl0, init_iters))
        kappa_candidate = k_old + smooth_plus(dl_final)
        return lax.cond(R0 <= 0.0, lambda _: k_old, lambda _: kappa_candidate, operand=None)

    def get_universal_kernel(self):
        def kernel(cell_sol_flat, X, cell_shape_grads, cell_JxW, cell_v_grads_JxW, kappa_elem):
            # Staggered: use internal kappa (kappa_elem) instead of kappa_tmp
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_u, cell_sol_phi = cell_sol_list
            cell_shape_grads_u = cell_shape_grads[:, :self.num_nodes_cumsum[1], :]
            cell_shape_grads_phi = cell_shape_grads[:, self.num_nodes_cumsum[1]:, :]
            cell_v_grads_JxW_u = cell_v_grads_JxW[:, :self.num_nodes_cumsum[1], :, :]
            cell_v_grads_JxW_phi = cell_v_grads_JxW[:, self.num_nodes_cumsum[1]:, :, :]
            cell_JxW_u, cell_JxW_phi = cell_JxW[0], cell_JxW[1]

            u_grads = jnp.sum(cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :], axis=1)
            F = u_grads + jnp.eye(self.dim)

            shape_vals_phi = self.fe_phi.shape_vals
            phi_quad = jnp.sum(shape_vals_phi[:, :, None] * cell_sol_phi[None, :, :], axis=1)
            phi_vals = phi_quad[:, 0]
            phi_vals_safe = jnp.maximum(phi_vals, 0.0)  # exclusively positive

            phi_grads = jnp.sum(
                cell_sol_phi[None, :, :, None] * cell_shape_grads_phi[:, :, None, :],
                axis=1,
            )

            # Staggered: use kappa from previous step (internal var), not kappa_tmp
            kappa = kappa_elem

            phi_grads_squeezed = jnp.squeeze(phi_grads, axis=1)
            P_int = jax.vmap(self.compute_PK1_base)(
                F, phi_grads_squeezed, phi_vals_safe, kappa
            )

            res_u = jnp.sum(P_int[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))

            grad_res = self.c_d * jnp.einsum("q1j,qn0j->n", phi_grads, cell_v_grads_JxW_phi)[:, None]

            phi_diff = phi_vals_safe - self.gamma_d * kappa
            penalty_term = phi_diff[:, None] * shape_vals_phi * cell_JxW_phi[:, None]
            penalty_res = self.beta_d * jnp.sum(penalty_term, axis=0)[:, None]

            weak_form = [res_u, grad_res + penalty_res]
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return kernel

    def _residual_with_bc_applied(self, res_list, sol_list, scale=1.0):
        """Same BC treatment as jax_fem solver: at Dirichlet dofs replace residual with (sol - bc_vals)*scale."""
        out = []
        for ind, fe in enumerate(self.fes):
            res = res_list[ind]
            sol = sol_list[ind]
            for i in range(len(fe.node_inds_list)):
                res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                    sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True
                )
                res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(
                    -fe.vals_list[i] * scale
                )
            out.append(res)
        return out

    def compute_residual(self, sol_list):
        res_list = super().compute_residual(sol_list)
        res_list_bc = self._residual_with_bc_applied(res_list, sol_list)
        norm_u = jnp.linalg.norm(res_list_bc[0])
        norm_phi = jnp.linalg.norm(res_list_bc[1])
        jax.debug.print(
            "global displacement (l2): {}  phi (l2): {}",
            norm_u,
            norm_phi,
        )
        return res_list

    @functools.partial(jax.jit, static_argnums=(0,))
    def local_update_kappa_field(self, u_sol, phi_sol, kappa_n):
        """Compute κ^(k+1) = local_update(u^(k), φ^(k), κ_n) without mutating state.

        Used by the inner staggered loop: takes current (u, phi) and previous
        converged load step kappa_n, returns the updated kappa field.
        JIT-compiled with self static so the kernel is compiled once per problem.
        """
        phi = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi, axis=-1)

        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)
        v_sed = jax.vmap(jax.vmap(self.psi_neohookean))(F)

        per_quad_update = jax.vmap(
            jax.vmap(self.local_update_kappa, in_axes=(0, 0, 0)),
            in_axes=(0, 0, 0),
        )
        return per_quad_update(v_sed, phi, kappa_n)

    def update_internal_vars(self, sol_list):
        old_kappa = self.internal_vars[0]
        u_sol, phi_sol = sol_list

        phi = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi, axis=-1)

        u_grad = self.fe_u.sol_to_grad(u_sol)
        I = jnp.eye(self.dim)
        F = u_grad + I

        v_sed = jax.vmap(jax.vmap(self.psi_neohookean))(F)

        per_quad_update = jax.vmap(
            jax.vmap(self.local_update_kappa, in_axes=(0, 0, 0)),
            in_axes=(0, 0, 0),
        )

        new_kappa = per_quad_update(v_sed, phi, old_kappa)
        self.internal_vars = [new_kappa]

    def get_trial_state_stats(self, sol_list):
        """Diagnostic: min/max of trial phi, kappa_tmp, and det(F) without mutating state."""
        u_sol, phi_sol = sol_list
        phi_quad = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi_quad, axis=-1)
        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)
        det_F = jnp.linalg.det(F)
        v_sed = jax.vmap(jax.vmap(self.psi_neohookean))(F)
        per_quad_update = jax.vmap(
            jax.vmap(self.local_update_kappa, in_axes=(0, 0, 0)),
            in_axes=(0, 0, 0),
        )
        kappa_elem = self.internal_vars[0]
        kappa_tmp = per_quad_update(v_sed, phi, kappa_elem)
        return {
            "phi_min": float(jnp.min(phi)),
            "phi_max": float(jnp.max(phi)),
            "kappa_tmp_min": float(jnp.min(kappa_tmp)),
            "kappa_tmp_max": float(jnp.max(kappa_tmp)),
            "det_F_min": float(jnp.min(det_F)),
            "det_F_max": float(jnp.max(det_F)),
        }

    def compute_cauchy_quads(self, u_sol, phi_sol):
        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)

        phi = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi, axis=-1)
        phi_grads = self.fe_phi.sol_to_grad(phi_sol)  # (num_cells, num_quads, 1, dim)

        kappa = self.internal_vars[0]
        sigma = jax.vmap(jax.vmap(self.compute_cauchy_stress))(F, phi_grads, phi, kappa)
        return sigma

    def compute_von_mises(self, u_sol, phi_sol):
        sigma = self.compute_cauchy_quads(u_sol, phi_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(self.dim)
        s_dev = sigma - hydrostatic
        von_mises = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))
        return jnp.mean(von_mises, axis=1)

    def compute_nodal_von_mises(self, u_sol, phi_sol):
        sigma = self.compute_cauchy_quads(u_sol, phi_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(self.dim)
        s_dev = sigma - hydrostatic
        von_mises_quad = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))
        return self.convert_quad_to_nodes(von_mises_quad)

    def compute_max_damage(self, sol_list):
        old_kappa = self.internal_vars[0].copy()
        u_sol, phi_sol = sol_list
        phi = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi, axis=-1)

        u_grad = self.fe_u.sol_to_grad(u_sol)
        I = jnp.eye(self.dim)
        F = u_grad + I

        v_sed = jax.vmap(jax.vmap(self.psi_neohookean))(F)

        per_quad_update = jax.vmap(
            jax.vmap(self.local_update_kappa, in_axes=(0, 0, 0)),
            in_axes=(0, 0, 0),
        )

        new_kappa = per_quad_update(v_sed, phi, old_kappa)
        damage = 1 - self.damage_function(new_kappa)
        return jnp.max(damage)

    def convert_quad_to_nodes(self, quad_data):
        """Convert quadrature point data to nodal data by interpolation.

        Args:
            quad_data: Array of shape (num_cells, num_quads) containing values at quadrature points

        Returns:
            nodal_data: Array of shape (num_nodes,) containing interpolated values at nodes
        """
        quad_data_np = np.array(quad_data)
        shape_vals = self.fe_u.shape_vals  # (num_quads, num_nodes_per_cell)
        cells = self.fe_u.cells  # (num_cells, num_nodes_per_cell), mesh node order
        num_nodes_total = self.fe_u.num_total_nodes

        cell_nodal_values = np.sum(
            quad_data_np[:, :, None] * shape_vals[None, :, :],
            axis=1
        )
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
