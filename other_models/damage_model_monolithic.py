"""
Monolithic damage: κ promoted to a third global FE unknown. No internal variables;
the consistency equation Rκ = sed + βd(φ−κ)/(η g(κ)) − κ is enforced in weak form.
"""

import numpy as np
import jax
import jax.numpy as jnp

from jax_fem.problem import Problem

jax.config.update("jax_enable_x64", True)

KAPPA_EPS = 1e-12
G_KAPPA_MIN = 1e-10
DET_F_MIN = 1e-12
SMOOTH_EPS = 1e-14


def smooth_lower(x, lower):
    """Smooth approximation to max(x, lower) with non-zero derivative everywhere."""
    d = x - lower
    return lower + 0.5 * (d + jnp.sqrt(d * d + SMOOTH_EPS))


class CoupledDamageProblemMonolithic(Problem):

    __hash__ = object.__hash__
    __eq__ = lambda _self, other: _self is other

    def custom_init(self, E, nu, kappa_thresh, eta, c_d, beta_d, local_only=False):
        self.fe_u = self.fes[0]
        self.fe_phi = self.fes[1]
        self.fe_kappa = self.fes[2]

        self.E, self.nu = E, nu
        self.mu = E / (2 * (1 + nu))
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

        self.kappa_threshold = kappa_thresh
        self.eta = eta
        self.local_only = bool(local_only)
        self.gamma_d = 0.0 if self.local_only else 1.0
        self.c_d = 0.0 if self.local_only else float(c_d)
        self.beta_d = beta_d
        self.eps_reg = 1e-10

        self.internal_vars = ()

    def damage_function(self, kappa):
        kappa_safe = smooth_lower(kappa, KAPPA_EPS)
        return jnp.exp(-self.eta * jnp.maximum(kappa_safe - self.kappa_threshold, 0.0))

    def _safe_det_and_log(self, F):
        det_F = jnp.linalg.det(F)
        det_F_safe = jnp.maximum(jnp.abs(det_F), DET_F_MIN)
        log_J = jnp.log(det_F_safe)
        return det_F_safe, log_J

    def psi_neohookean(self, F):
        _, log_J = self._safe_det_and_log(F)
        C = F @ F.T
        I1 = jnp.trace(C)
        return (self.mu / 2.0) * (I1 - 3.0) - self.mu * log_J + (self.lmbda / 2.0) * log_J ** 2

    def compute_PK1_base(self, F, phi_grad, phi, kappa):
        _, log_J = self._safe_det_and_log(F)
        F_inv_T = jnp.linalg.inv(F).T
        P_undamaged = self.mu * F + (self.lmbda * log_J - self.mu) * F_inv_T
        g = self.damage_function(smooth_lower(kappa, KAPPA_EPS))
        return g * P_undamaged

    def compute_cauchy_stress(self, F, phi_grad, phi, kappa):
        P = self.compute_PK1_base(F, phi_grad, phi, kappa)
        det_F_safe, _ = self._safe_det_and_log(F)
        return jnp.matmul(P, jnp.swapaxes(F, -1, -2)) / det_F_safe

    def residual_kappa_quad(self, sed, phi_val, kappa_val):
        """Scalar Rκ at a quad: sed + βd(φ−κ)/(η g(κ)) − κ, or local_only: sed − κ."""
        if self.local_only:
            return sed - kappa_val
        kappa_safe = smooth_lower(kappa_val, KAPPA_EPS)
        g_kappa = self.damage_function(kappa_safe)
        g_kappa_safe = smooth_lower(g_kappa, G_KAPPA_MIN)
        denom = self.eta * g_kappa_safe
        return (
            sed
            + self.beta_d * (phi_val - kappa_val) / denom
            - kappa_val
        )

    def get_universal_kernel(self):
        n0, n1, n2 = (
            self.num_nodes_cumsum[0],
            self.num_nodes_cumsum[1],
            self.num_nodes_cumsum[2],
        )
        n3 = self.num_nodes_cumsum[3]

        def kernel(cell_sol_flat, _X, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_u, cell_sol_phi, cell_sol_kappa = cell_sol_list

            cell_shape_grads_u = cell_shape_grads[:, n0:n1, :]
            cell_shape_grads_phi = cell_shape_grads[:, n1:n2, :]
            cell_shape_grads_kappa = cell_shape_grads[:, n2:n3, :]
            cell_v_grads_JxW_u = cell_v_grads_JxW[:, n0:n1, :, :]
            cell_v_grads_JxW_phi = cell_v_grads_JxW[:, n1:n2, :, :]
            cell_JxW_phi = cell_JxW[1]
            cell_JxW_kappa = cell_JxW[2]

            u_grads = jnp.sum(
                cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :], axis=1
            )
            F = u_grads + jnp.eye(self.dim)
            sed_quads = jax.vmap(self.psi_neohookean)(F)

            shape_vals_phi = self.fe_phi.shape_vals
            phi_quad = jnp.sum(
                shape_vals_phi[:, :, None] * cell_sol_phi[None, :, :], axis=1
            )
            phi_vals = phi_quad[:, 0]
            phi_vals_safe = smooth_lower(phi_vals, 0.0)

            phi_grads = jnp.sum(
                cell_sol_phi[None, :, :, None] * cell_shape_grads_phi[:, :, None, :],
                axis=1,
            )

            shape_vals_kappa = self.fe_kappa.shape_vals
            kappa_quad = jnp.sum(
                shape_vals_kappa[:, :, None] * cell_sol_kappa[None, :, :], axis=1
            )
            kappa_vals = smooth_lower(kappa_quad[:, 0], KAPPA_EPS)

            phi_grads_squeezed = jnp.squeeze(phi_grads, axis=1)
            P_int = jax.vmap(self.compute_PK1_base)(
                F, phi_grads_squeezed, phi_vals_safe, kappa_vals
            )
            res_u = jnp.sum(P_int[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))

            grad_res = self.c_d * jnp.einsum(
                "q1j,qn0j->n", phi_grads, cell_v_grads_JxW_phi
            )[:, None]
            phi_diff = phi_vals_safe - self.gamma_d * kappa_vals
            penalty_term = phi_diff[:, None] * shape_vals_phi * cell_JxW_phi[:, None]
            penalty_res = self.beta_d * jnp.sum(penalty_term, axis=0)[:, None]
            res_phi = grad_res + penalty_res

            R_kappa = jax.vmap(self.residual_kappa_quad)(
                sed_quads, phi_vals_safe, kappa_vals
            )
            kappa_term = (
                R_kappa[:, None] * shape_vals_kappa * cell_JxW_kappa[:, None]
            )
            res_kappa = jnp.sum(kappa_term, axis=0)[:, None]

            weak_form = [res_u, res_phi, res_kappa]
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return kernel

    def _residual_with_bc_applied(self, res_list, sol_list, scale=1.0):
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
        norm_kappa = jnp.linalg.norm(res_list_bc[2])
        jax.debug.print(
            "global displacement (l2): {}  phi (l2): {}  kappa (l2): {}",
            norm_u,
            norm_phi,
            norm_kappa,
        )
        return res_list

    def compute_cauchy_quads(self, u_sol, phi_sol, kappa_sol):
        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)

        phi = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi, axis=-1)
        phi_grads = self.fe_phi.sol_to_grad(phi_sol)

        kappa = self.fe_kappa.convert_from_dof_to_quad(kappa_sol)
        kappa = jnp.squeeze(kappa, axis=-1)
        kappa = smooth_lower(kappa, KAPPA_EPS)

        sigma = jax.vmap(jax.vmap(self.compute_cauchy_stress))(
            F, phi_grads, phi, kappa
        )
        return sigma

    def compute_von_mises(self, u_sol, phi_sol, kappa_sol):
        sigma = self.compute_cauchy_quads(u_sol, phi_sol, kappa_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(self.dim)
        s_dev = sigma - hydrostatic
        von_mises = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))
        return jnp.mean(von_mises, axis=1)

    def compute_nodal_von_mises(self, u_sol, phi_sol, kappa_sol):
        sigma = self.compute_cauchy_quads(u_sol, phi_sol, kappa_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(self.dim)
        s_dev = sigma - hydrostatic
        von_mises_quad = jnp.sqrt(
            3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1))
        )
        return self.convert_quad_to_nodes(von_mises_quad)

    def compute_max_damage(self, sol_list):
        kappa_sol = sol_list[2]
        kappa = self.fe_kappa.convert_from_dof_to_quad(kappa_sol)
        kappa = jnp.squeeze(kappa, axis=-1)
        kappa = smooth_lower(kappa, KAPPA_EPS)
        damage = 1 - self.damage_function(kappa)
        return jnp.max(damage)

    def convert_quad_to_nodes(self, quad_data):
        quad_data_np = np.array(quad_data)
        shape_vals = self.fe_u.shape_vals
        cells = self.fe_u.cells
        num_nodes_total = self.fe_u.num_total_nodes

        cell_nodal_values = np.sum(
            quad_data_np[:, :, None] * shape_vals[None, :, :], axis=1
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
