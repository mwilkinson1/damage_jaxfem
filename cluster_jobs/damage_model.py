import jax
import jax.numpy as jnp

from jax_fem.problem import Problem
from custom_vjp_kkt import kappa_update_kkt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

DET_F_MIN = 1e-12

# Safe positive/negative parts for autodiff (avoids kinks from max(x,0)/min(x,0) at 0)
def smooth_plus(x, eps=1e-8):
    return 0.5 * (x + jnp.sqrt(x*x + eps*eps))

class CoupledDamageProblem(Problem):
    
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
        
        # Internal variables
        num_cells = self.fe_u.num_cells
        num_quads = self.fe_u.num_quads
        self.internal_vars = [jnp.zeros((num_cells, num_quads))]

    def damage_function(self, kappa):
        return jnp.exp(-self.eta * smooth_plus(kappa - self.kappa_threshold))

    def psi_neohookean(self, F):
        """Neo-Hookean strain energy for F of shape (3,3)."""
        det_F = jnp.linalg.det(F)
        det_F_safe = jnp.maximum(jnp.abs(det_F), DET_F_MIN)
        C = F @ F.T  # (3,3)
        I1 = jnp.trace(C)
        log_J = jnp.log(det_F_safe)
        psi_val = (self.mu / 2.0) * (I1 - 3.0) - self.mu * log_J + (self.lmbda / 2.0) * (log_J ** 2)
        return psi_val  # scalar

    def psi_grad_fn(self, phi_grad, F):
        """Nonlocal gradient energy term: (c_d/2) * |F^{-T} grad_X phi|^2"""
        # phi_grad is (3,) (already squeezed to vector form)
        # F is (3,3)
        F_inv_T = jnp.linalg.inv(F).T      # (3,3)
        g = F_inv_T @ phi_grad             # (3,)
        return 0.5 * self.c_d * jnp.dot(g, g)  # scalar

    def psi_plty_fn(self, phi, kappa):
        # phi and kappa are both (1,) or float
        return 0.5 * self.beta_d * (phi - kappa) ** 2

    def psi_int(self, F, phi, phi_grad, kappa):
        """Internal strain energy density as formulated from the paper.

        Psi_int(F, phi, grad_X phi, kappa)
            = f_d(kappa) * psi_e(F)
            + psi_nloc^grad(grad_X phi; F)
            + psi_nloc^plty(phi, kappa)

        psi_e(F): strain energy -- implemented as self.psi_neohookean(F)
        psi_grad_fn(grad_X phi, F): gradient (nonlocal) part
        psi_plty_fn(phi, kappa): penalty (proximity, history) part
        """
        damage = self.damage_function(kappa)
        psi_e = self.psi_neohookean(F)
        psi_grad = self.psi_grad_fn(phi_grad, F)
        psi_plty = self.psi_plty_fn(phi, kappa)
        return damage * psi_e + psi_grad + psi_plty

    def compute_PK1_base(self, F, phi_grad, phi, kappa):
        """Compute the first Piola-Kirchhoff (PK1) stress tensor using automatic differentiation (condensed version)."""
        phi_grad_X = (
            jnp.squeeze(phi_grad, axis=-2)
            if phi_grad.ndim >= 2 and phi_grad.shape[-2] == 1
            else phi_grad
        )
        P = jax.grad(lambda FF: self.psi_int(FF, phi, phi_grad_X, kappa))(F)
        return P

    def compute_cauchy_stress(self, F, phi_grad, phi, kappa):
        """Cauchy stress sigma = (1/J) P F^T from PK1."""
        P = self.compute_PK1_base(F, phi_grad, phi, kappa)
        det_F = jnp.linalg.det(F)
        det_F_safe = jnp.maximum(jnp.abs(det_F), DET_F_MIN)
        sigma = jnp.matmul(P, jnp.swapaxes(F, -1, -2)) / det_F_safe
        return sigma

    @staticmethod
    def von_mises_from_cauchy(sigma):
        """Von Mises stress from Cauchy stress tensor; sigma shape (..., dim, dim)."""
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(sigma.shape[-1])
        s_dev = sigma - hydrostatic
        return jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))

    def compute_cauchy_quads(self, u_sol, phi_sol):
        """Cauchy stress at all quadrature points; shape (num_cells, num_quads, dim, dim)."""
        u_grad = self.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(self.dim)
        phi = self.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi = jnp.squeeze(phi, axis=-1)
        phi_grads = self.fe_phi.sol_to_grad(phi_sol)
        kappa = self.internal_vars[0]
        return jax.vmap(jax.vmap(self.compute_cauchy_stress))(F, phi_grads, phi, kappa)

    def compute_von_mises_quads(self, u_sol, phi_sol):
        """Von Mises stress at all quadrature points; shape (num_cells, num_quads)."""
        sigma = self.compute_cauchy_quads(u_sol, phi_sol)
        return self.von_mises_from_cauchy(sigma)

    def local_update_kappa(self, sed, phi_val, k_old):
        phi_kappa_scale = 0.0 if self.local_only else 1.0
        return kappa_update_kkt(
            sed,
            phi_val,
            k_old,
            self.kappa_threshold,
            self.eta,
            self.beta_d,
            eps_sp=1e-8,
            newton_tol=1e-8,
            max_iters=100,
            phi_kappa_scale=phi_kappa_scale,
        )

    def get_universal_kernel(self):
        grad_psi_wrt_phi_grad = jax.grad(self.psi_grad_fn, argnums=0)
        grad_psi_wrt_phi = jax.grad(self.psi_plty_fn, argnums=0)

        def kernel(cell_sol_flat, X, cell_shape_grads, cell_JxW, cell_v_grads_JxW, kappa_elem):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # X: (num_quads, dim)
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_JxW: (num_vars, num_quads)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)
            
            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol_u, cell_sol_phi = cell_sol_list
            cell_shape_grads_u = cell_shape_grads[:, :self.num_nodes_cumsum[1], :]
            cell_shape_grads_phi = cell_shape_grads[:, self.num_nodes_cumsum[1]:, :]
            cell_v_grads_JxW_u = cell_v_grads_JxW[:, :self.num_nodes_cumsum[1], :, :]
            cell_v_grads_JxW_phi = cell_v_grads_JxW[:, self.num_nodes_cumsum[1]:, :, :]
            cell_JxW_phi = cell_JxW[1]
            
            u_grads = jnp.sum(cell_sol_u[None, :, :, None] * cell_shape_grads_u[:, :, None, :], axis=1)
            F = u_grads + jnp.eye(self.dim)
            
            shape_vals_phi = self.fe_phi.shape_vals
            phi_quad = jnp.sum(shape_vals_phi[:, :, None] * cell_sol_phi[None, :, :], axis=1)
            phi_vals = phi_quad[:, 0]

            phi_grads = jnp.sum(
                cell_sol_phi[None, :, :, None] * cell_shape_grads_phi[:, :, None, :],
                axis=1,
            )

            sed_quads = jax.vmap(self.psi_neohookean)(F)
            kappa_tmp = jax.vmap(self.local_update_kappa, in_axes=(0, 0, 0))(
                sed_quads, phi_vals, kappa_elem
            )

            phi_grads_squeezed = jnp.squeeze(phi_grads, axis=1)
            P_int = jax.vmap(self.compute_PK1_base)(
                F, phi_grads_squeezed, phi_vals, kappa_tmp
            )

            res_u = jnp.sum(P_int[:, None, :, :] * cell_v_grads_JxW_u, axis=(0, -1))

            grad_energy_flux = jax.vmap(grad_psi_wrt_phi_grad)(phi_grads_squeezed, F)
            grad_res = jnp.einsum(
                "qi,qni->n",
                grad_energy_flux,
                cell_v_grads_JxW_phi[:, :, 0, :],
            )[:, None]

            kappa_target = self.gamma_d * kappa_tmp
            penalty_drive = jax.vmap(grad_psi_wrt_phi)(phi_vals, kappa_target)
            penalty_res = jnp.sum(
                penalty_drive[:, None] * shape_vals_phi * cell_JxW_phi[:, None],
                axis=0,
            )[:, None]

            weak_form = [res_u, grad_res + penalty_res]
            return jax.flatten_util.ravel_pytree(weak_form)[0]

        return kernel

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


def get_boundary_force_component(problem, sol_list, fe_index, bc_group_index):
    """Reaction force (same magnitude, opposite sign to internal force) for one Dirichlet BC group.

    The raw residual at Dirichlet dofs equals the internal force at those dofs.
    Reaction force from the support = -residual, so summing over the group gives
    the total reaction force in that component on that boundary.

    Parameters
    ----------
    problem : Problem
        The coupled problem (must have compute_residual).
    sol_list : list of arrays
        Solution list [u_sol, phi_sol, ...].
    fe_index : int
        Which FE variable (0 = displacement).
    bc_group_index : int
        Which Dirichlet BC group in node_inds_list / vec_inds_list (0 = first group).

    Returns
    -------
    float
        Total reaction force in the component enforced by that BC group (e.g. X force on left).
    """
    res_list = problem.compute_residual(sol_list)
    fe = problem.fes[fe_index]
    node_inds = fe.node_inds_list[bc_group_index]
    vec_inds = fe.vec_inds_list[bc_group_index]
    residual_at_dofs = res_list[fe_index][node_inds, vec_inds]
    return float(-jnp.sum(residual_at_dofs))

