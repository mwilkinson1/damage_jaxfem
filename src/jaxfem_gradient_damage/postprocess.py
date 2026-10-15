import jax.numpy as jnp
import numpy as np

from jaxfem_gradient_damage.constitutive import (
    cauchy_stress,
    von_mises_stress,
)


def quadrature_to_nodes(quadrature_values, finite_element):
    values = np.asarray(quadrature_values)
    shape_values = np.asarray(finite_element.shape_vals)
    cells = np.asarray(finite_element.cells)
    cell_node_values = np.sum(
        values[:, :, None] * shape_values[None, :, :],
        axis=1,
    )
    nodal_sum = np.zeros(finite_element.num_total_nodes)
    nodal_count = np.zeros(
        finite_element.num_total_nodes,
        dtype=np.int32,
    )
    np.add.at(nodal_sum, cells.reshape(-1), cell_node_values.reshape(-1))
    np.add.at(nodal_count, cells.reshape(-1), 1)
    nodal_values = np.divide(
        nodal_sum,
        nodal_count,
        out=np.zeros_like(nodal_sum),
        where=nodal_count > 0,
    )
    return jnp.asarray(nodal_values)


def boundary_reaction_component(
    problem,
    solution,
    field_index,
    boundary_condition_index,
):
    residual = problem.compute_residual(solution)
    finite_element = problem.fes[field_index]
    node_indices = finite_element.node_inds_list[boundary_condition_index]
    vector_indices = finite_element.vec_inds_list[
        boundary_condition_index
    ]
    return -jnp.sum(residual[field_index][node_indices, vector_indices])


def stresses_from_first_piola(first_piola, deformation_gradient):
    cauchy = cauchy_stress(first_piola, deformation_gradient)
    return cauchy, von_mises_stress(cauchy)
