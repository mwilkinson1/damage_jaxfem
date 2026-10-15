import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.solver import solver

from jaxfem_gradient_damage import (
    DamageParameters,
    ElasticParameters,
    LocalDamageProblem,
    ModelParameters,
)
from jaxfem_gradient_damage.stepping import (
    StepControl,
    run_adaptive_steps,
)


jax.config.update("jax_enable_x64", True)


def zero(_point):
    return 0.0


def moving_x(value):
    return lambda _point: value


def create_mesh():
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


def create_boundary_conditions(displacement):
    left = lambda point: jnp.isclose(point[0], 0.0)
    right = lambda point: jnp.isclose(point[0], 1.0)
    bottom = lambda point: jnp.isclose(point[1], 0.0)
    front = lambda point: jnp.isclose(point[2], 0.0)
    return [
        [left, right, bottom, front],
        [0, 0, 1, 2],
        [zero, moving_x(displacement), zero, zero],
    ]


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def run_case(arguments):
    boundary_conditions = create_boundary_conditions(0.0)
    model_parameters = ModelParameters(
        elastic=ElasticParameters(42.0, 0.45),
        damage=DamageParameters(
            threshold=arguments.kappa_threshold,
            saturation=arguments.eta,
        ),
    )
    problem = LocalDamageProblem(
        create_mesh(),
        vec=3,
        dim=3,
        ele_type="HEX8",
        dirichlet_bc_info=boundary_conditions,
        gauss_order=3,
        additional_info=(model_parameters,),
    )
    solver_options = {
        "petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"},
        "tol": 1e-8,
        "rel_tol": 1e-8,
    }
    records = []

    def solve_step(load, previous_solution):
        boundary_conditions[-1][1] = moving_x(load)
        problem.fes[0].update_Dirichlet_boundary_conditions(
            boundary_conditions
        )
        options = dict(solver_options)
        if previous_solution is not None:
            options["initial_guess"] = previous_solution
        return solver(problem, solver_options=options)

    def accept_step(load, solution):
        problem.update_internal_vars(solution)
        cauchy = problem.cauchy_stress_quadrature(solution[0])
        records.append(
            {
                "prescribed_displacement": load,
                "stretch": 1.0 + load,
                "sigma_11": float(jnp.mean(cauchy[..., 0, 0])),
                "kappa": float(jnp.mean(problem.internal_vars[0])),
                "damage": float(jnp.mean(problem.damage())),
            }
        )
        write_records(arguments.output, records)

    control = StepControl(
        initial_increment=arguments.initial_increment,
        minimum_increment=arguments.minimum_increment,
        maximum_increment=arguments.maximum_increment,
        maximum_runtime_seconds=arguments.max_runtime,
    )
    result = run_adaptive_steps(
        arguments.maximum_displacement,
        control,
        solve_step,
        accept_step,
    )
    print(
        f"stop_reason={result.stop_reason}, load={result.load:.6f}, "
        f"accepted={result.accepted_steps}, rejected={result.rejected_steps}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--kappa-threshold", type=float, default=0.2)
    parser.add_argument("--maximum-displacement", type=float, default=0.5)
    parser.add_argument("--initial-increment", type=float, default=0.005)
    parser.add_argument("--minimum-increment", type=float, default=1e-5)
    parser.add_argument("--maximum-increment", type=float, default=0.01)
    parser.add_argument("--max-runtime", type=float, default=285.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figure_01/single_element.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_case(parse_arguments())
