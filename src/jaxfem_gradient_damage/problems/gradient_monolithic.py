import jax
import jax.numpy as jnp

from jax_fem.problem import Problem

from jaxfem_gradient_damage.constitutive import (
    cauchy_stress,
    damage_degradation,
    first_piola_stress,
    neo_hookean_energy,
    nonlocal_flux,
    penalty_force,
)
from jaxfem_gradient_damage.kkt import update_local_history


class MonolithicGradientDamageProblem(Problem):
    def custom_init(self, model_parameters):
        self.parameters = model_parameters
        self.displacement_element = self.fes[0]
        self.nonlocal_element = self.fes[1]
        history_shape = (
            self.displacement_element.num_cells,
            self.displacement_element.num_quads,
        )
        self.internal_vars = [jnp.zeros(history_shape)]

    def update_history_point(
        self,
        elastic_energy,
        nonlocal_history,
        previous_history,
    ):
        return update_local_history(
            elastic_energy,
            nonlocal_history,
            previous_history,
            self.parameters.damage,
            self.parameters.local_solver,
        )

    def kernel_history_point(
        self,
        elastic_energy,
        nonlocal_history,
        previous_history,
    ):
        return self.update_history_point(
            elastic_energy,
            nonlocal_history,
            previous_history,
        )

    def get_universal_kernel(self):
        elastic_parameters = self.parameters.elastic
        damage_parameters = self.parameters.damage
        displacement_nodes = self.num_nodes_cumsum[1]
        nonlocal_shape_values = self.nonlocal_element.shape_vals

        def kernel(
            cell_solution,
            _coordinates,
            shape_gradients,
            quadrature_weights,
            test_gradients,
            previous_history,
        ):
            displacement, nonlocal_history_nodes = (
                self.unflatten_fn_dof(cell_solution)
            )
            displacement_shape_gradients = shape_gradients[
                :, :displacement_nodes, :
            ]
            nonlocal_shape_gradients = shape_gradients[
                :, displacement_nodes:, :
            ]
            displacement_test_gradients = test_gradients[
                :, :displacement_nodes, :, :
            ]
            nonlocal_test_gradients = test_gradients[
                :, displacement_nodes:, 0, :
            ]
            displacement_gradients = jnp.sum(
                displacement[None, :, :, None]
                * displacement_shape_gradients[:, :, None, :],
                axis=1,
            )
            deformation_gradients = (
                displacement_gradients + jnp.eye(self.dim)
            )
            nonlocal_history = jnp.sum(
                nonlocal_shape_values[:, :, None]
                * nonlocal_history_nodes[None, :, :],
                axis=1,
            )[:, 0]
            nonlocal_gradients = jnp.sum(
                nonlocal_history_nodes[None, :, :, None]
                * nonlocal_shape_gradients[:, :, None, :],
                axis=1,
            )[:, 0, :]
            elastic_energies = jax.vmap(
                lambda gradient: neo_hookean_energy(
                    gradient,
                    elastic_parameters,
                )
            )(deformation_gradients)
            trial_history = jax.vmap(self.kernel_history_point)(
                elastic_energies,
                nonlocal_history,
                previous_history,
            )
            first_piola = jax.vmap(
                lambda gradient, phi, phi_gradient, kappa:
                first_piola_stress(
                    gradient,
                    phi,
                    phi_gradient,
                    kappa,
                    elastic_parameters,
                    damage_parameters,
                )
            )(
                deformation_gradients,
                nonlocal_history,
                nonlocal_gradients,
                trial_history,
            )
            displacement_residual = jnp.sum(
                first_piola[:, None, :, :]
                * displacement_test_gradients,
                axis=(0, -1),
            )
            flux = jax.vmap(
                lambda gradient, deformation_gradient: nonlocal_flux(
                    deformation_gradient,
                    gradient,
                    damage_parameters,
                )
            )(nonlocal_gradients, deformation_gradients)
            gradient_residual = jnp.einsum(
                "qi,qni->n",
                flux,
                nonlocal_test_gradients,
            )[:, None]
            penalty = jax.vmap(
                lambda phi, kappa: penalty_force(
                    phi,
                    kappa,
                    damage_parameters,
                )
            )(nonlocal_history, trial_history)
            penalty_residual = jnp.sum(
                penalty[:, None]
                * nonlocal_shape_values
                * quadrature_weights[1, :, None],
                axis=0,
            )[:, None]
            residual = [
                displacement_residual,
                gradient_residual + penalty_residual,
            ]
            return jax.flatten_util.ravel_pytree(residual)[0]

        return kernel

    def trial_history(self, solution):
        displacement, nonlocal_solution = solution
        deformation_gradients = (
            self.displacement_element.sol_to_grad(displacement)
            + jnp.eye(self.dim)
        )
        elastic_energies = jax.vmap(
            jax.vmap(
                lambda gradient: neo_hookean_energy(
                    gradient,
                    self.parameters.elastic,
                )
            )
        )(deformation_gradients)
        nonlocal_history = self.nonlocal_element.convert_from_dof_to_quad(
            nonlocal_solution
        )[..., 0]
        update_cells = jax.vmap(
            jax.vmap(self.update_history_point)
        )
        return update_cells(
            elastic_energies,
            nonlocal_history,
            self.internal_vars[0],
        )

    def update_internal_vars(self, solution):
        self.internal_vars = [self.trial_history(solution)]

    def damage(self):
        return 1.0 - damage_degradation(
            self.internal_vars[0],
            self.parameters.damage,
        )

    def cauchy_stress_quadrature(self, solution):
        displacement, nonlocal_solution = solution
        deformation_gradient = (
            self.displacement_element.sol_to_grad(displacement)
            + jnp.eye(self.dim)
        )
        nonlocal_history = self.nonlocal_element.convert_from_dof_to_quad(
            nonlocal_solution
        )[..., 0]
        nonlocal_gradient = self.nonlocal_element.sol_to_grad(
            nonlocal_solution
        )[..., 0, :]
        first_piola = jax.vmap(
            jax.vmap(
                lambda gradient, phi, phi_gradient, kappa:
                first_piola_stress(
                    gradient,
                    phi,
                    phi_gradient,
                    kappa,
                    self.parameters.elastic,
                    self.parameters.damage,
                )
            )
        )(
            deformation_gradient,
            nonlocal_history,
            nonlocal_gradient,
            self.internal_vars[0],
        )
        return jax.vmap(jax.vmap(cauchy_stress))(
            first_piola,
            deformation_gradient,
        )
