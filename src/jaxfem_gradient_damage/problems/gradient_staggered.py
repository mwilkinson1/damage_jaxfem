import jax
import jax.numpy as jnp

from jaxfem_gradient_damage.constitutive import neo_hookean_energy
from jaxfem_gradient_damage.problems.gradient_monolithic import (
    MonolithicGradientDamageProblem,
)


class StaggeredGradientDamageProblem(MonolithicGradientDamageProblem):
    def kernel_history_point(
        self,
        _elastic_energy,
        _nonlocal_history,
        previous_history,
    ):
        return previous_history

    def history_candidate(self, solution, previous_load_history):
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
            previous_load_history,
        )

    def update_internal_vars(self, solution):
        previous_history = self.internal_vars[0]
        self.internal_vars = [
            self.history_candidate(solution, previous_history)
        ]
