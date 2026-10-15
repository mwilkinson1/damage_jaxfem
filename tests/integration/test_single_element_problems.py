import jax.numpy as jnp
import jax
import numpy as np
import pytest

from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.solver import solver

from jaxfem_gradient_damage import (
    DamageParameters,
    ElasticParameters,
    LocalDamageProblem,
    ModelParameters,
    MonolithicGradientDamageProblem,
    StaggeredGradientDamageProblem,
)


def zero(_point):
    return 0.0


def prescribed_x(value):
    return lambda _point: value


def left(point):
    return jnp.isclose(point[0], 0.0)


def right(point):
    return jnp.isclose(point[0], 1.0)


def bottom(point):
    return jnp.isclose(point[1], 0.0)


def front(point):
    return jnp.isclose(point[2], 0.0)


def mesh():
    mesh_data = box_mesh(
        Nx=1,
        Ny=1,
        Nz=1,
        domain_x=1.0,
        domain_y=1.0,
        domain_z=1.0,
    )
    return Mesh(
        mesh_data.points,
        mesh_data.cells_dict["hexahedron"],
        ele_type="HEX8",
    )


def displacement_boundary_conditions(value):
    return [
        [left, right, bottom, front],
        [0, 0, 1, 2],
        [zero, prescribed_x(value), zero, zero],
    ]


def parameters():
    return ModelParameters(
        elastic=ElasticParameters(42.0, 0.45),
        damage=DamageParameters(
            threshold=0.2,
            saturation=2.0,
            gradient_coefficient=1.0,
            penalty_coefficient=100.0,
        ),
    )


SOLVER_OPTIONS = {
    "umfpack_solver": {},
    "tol": 1e-8,
    "rel_tol": 1e-8,
}


@pytest.mark.integration
def test_local_problem_one_increment():
    element_mesh = mesh()
    problem = LocalDamageProblem(
        element_mesh,
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=displacement_boundary_conditions(0.01),
        gauss_order=2,
        additional_info=(parameters(),),
    )

    solution = solver(problem, solver_options=SOLVER_OPTIONS)
    problem.update_internal_vars(solution)

    assert np.all(np.isfinite(np.asarray(solution[0])))
    assert np.all(np.asarray(problem.internal_vars[0]) >= 0.0)


@pytest.mark.integration
@pytest.mark.parametrize(
    "problem_type",
    [MonolithicGradientDamageProblem, StaggeredGradientDamageProblem],
)
def test_gradient_problem_one_increment(problem_type):
    element_mesh = mesh()
    problem = problem_type(
        [element_mesh, element_mesh],
        vec=[3, 1],
        dim=3,
        ele_type=["HEX8", "HEX8"],
        dirichlet_bc_info=[
            displacement_boundary_conditions(0.01),
            None,
        ],
        gauss_order=[2, 2],
        additional_info=(parameters(),),
    )

    solution = solver(problem, solver_options=SOLVER_OPTIONS)
    problem.update_internal_vars(solution)

    assert np.all(np.isfinite(np.asarray(solution[0])))
    assert np.all(np.isfinite(np.asarray(solution[1])))
    assert np.all(np.isfinite(np.asarray(problem.internal_vars[0])))


@pytest.mark.integration
def test_monolithic_element_jvp_matches_finite_difference():
    element_mesh = mesh()
    problem = MonolithicGradientDamageProblem(
        [element_mesh, element_mesh],
        vec=[3, 1],
        dim=3,
        ele_type=["HEX8", "HEX8"],
        dirichlet_bc_info=[
            displacement_boundary_conditions(0.0),
            None,
        ],
        gauss_order=[2, 2],
        additional_info=(parameters(),),
    )
    kernel = problem.get_universal_kernel()
    cell_solution = jnp.concatenate(
        (jnp.zeros(24), 0.5 * jnp.ones(8))
    )
    direction = jnp.linspace(
        -0.05,
        0.05,
        problem.num_total_dofs_all_vars,
    )
    arguments = (
        problem.physical_quad_points[0],
        problem.shape_grads[0],
        problem.JxW[0],
        problem.v_grads_JxW[0],
        problem.internal_vars[0][0],
    )
    residual = lambda values: kernel(values, *arguments)
    _, automatic = jax.jvp(
        residual,
        (cell_solution,),
        (direction,),
    )
    step = 1e-6
    finite_difference = (
        residual(cell_solution + step * direction)
        - residual(cell_solution - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        automatic,
        finite_difference,
        rtol=1e-5,
        atol=1e-7,
    )
