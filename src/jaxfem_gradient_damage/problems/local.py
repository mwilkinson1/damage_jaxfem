import jax
import jax.numpy as jnp

from jax_fem.problem import Problem

from jaxfem_gradient_damage.constitutive import (
    cauchy_stress,
    damage_degradation,
    degraded_first_piola_stress,
    neo_hookean_energy,
)
from jaxfem_gradient_damage.kkt import update_local_history_only


class LocalDamageProblem(Problem):
    def custom_init(self, model_parameters):
        self.parameters = model_parameters
        self.displacement_element = self.fes[0]
        history_shape = (
            self.displacement_element.num_cells,
            self.displacement_element.num_quads,
        )
        self.internal_vars = [jnp.zeros(history_shape)]

    def get_universal_kernel(self):
        elastic_parameters = self.parameters.elastic
        damage_parameters = self.parameters.damage
        finite_element = self.displacement_element

        def kernel(
            cell_solution,
            _coordinates,
            shape_gradients,
            _quadrature_weights,
            test_gradients,
            previous_history,
        ):
            displacement = self.unflatten_fn_dof(cell_solution)[0]
            displacement_gradients = jnp.sum(
                displacement[None, :, :, None]
                * shape_gradients[:, :, None, :],
                axis=1,
            )
            deformation_gradients = (
                displacement_gradients + jnp.eye(self.dim)
            )
            elastic_energies = jax.vmap(
                lambda deformation_gradient: neo_hookean_energy(
                    deformation_gradient,
                    elastic_parameters,
                )
            )(deformation_gradients)
            trial_history = jax.vmap(update_local_history_only)(
                elastic_energies,
                previous_history,
            )
            first_piola = jax.vmap(
                lambda deformation_gradient, local_history:
                degraded_first_piola_stress(
                    deformation_gradient,
                    local_history,
                    elastic_parameters,
                    damage_parameters,
                )
            )(deformation_gradients, trial_history)
            residual = jnp.sum(
                first_piola[:, None, :, :] * test_gradients,
                axis=(0, -1),
            )
            return jax.flatten_util.ravel_pytree([residual])[0]

        return kernel

    def trial_history(self, displacement):
        displacement_gradient = self.displacement_element.sol_to_grad(
            displacement
        )
        deformation_gradient = displacement_gradient + jnp.eye(self.dim)
        elastic_energy = jax.vmap(
            jax.vmap(
                lambda gradient: neo_hookean_energy(
                    gradient,
                    self.parameters.elastic,
                )
            )
        )(deformation_gradient)
        return jnp.maximum(self.internal_vars[0], elastic_energy)

    def update_internal_vars(self, solution):
        displacement = solution[0]
        self.internal_vars = [self.trial_history(displacement)]

    def damage(self):
        return 1.0 - damage_degradation(
            self.internal_vars[0],
            self.parameters.damage,
        )

    def cauchy_stress_quadrature(self, displacement):
        deformation_gradient = (
            self.displacement_element.sol_to_grad(displacement)
            + jnp.eye(self.dim)
        )
        first_piola = jax.vmap(
            jax.vmap(
                lambda gradient, history: degraded_first_piola_stress(
                    gradient,
                    history,
                    self.parameters.elastic,
                    self.parameters.damage,
                )
            )
        )(deformation_gradient, self.internal_vars[0])
        return jax.vmap(jax.vmap(cauchy_stress))(
            first_piola,
            deformation_gradient,
        )
