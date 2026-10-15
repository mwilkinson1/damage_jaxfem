import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver

from jaxfem_gradient_damage import (
    DamageParameters,
    ElasticParameters,
    ModelParameters,
    MonolithicGradientDamageProblem,
)
from jaxfem_gradient_damage.constitutive import von_mises_stress
from jaxfem_gradient_damage.stepping import (
    StepControl,
    run_adaptive_steps,
)


jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[3]


def zero(_point):
    return 0.0


def moving_x(value):
    return lambda _point: value


def boundary_conditions(displacement):
    left = lambda point: jnp.isclose(point[0], -100.0)
    right = lambda point: jnp.isclose(point[0], 100.0)
    return [
        [[left, left, left, right][index] for index in range(4)],
        [0, 1, 2, 0],
        [zero, zero, zero, moving_x(displacement)],
    ]


def create_problem(arguments, conditions):
    mesh_path = (
        ROOT
        / "meshes"
        / f"notched_plate_C3D20_{arguments.mesh}.inp"
    )
    mesh_data = meshio.read(mesh_path)
    mesh = Mesh(
        mesh_data.points,
        mesh_data.cells_dict["hexahedron20"],
        ele_type="HEX20",
    )
    parameters = ModelParameters(
        elastic=ElasticParameters(arguments.young_modulus, 0.3),
        damage=DamageParameters(
            threshold=0.1,
            saturation=0.001,
            gradient_coefficient=1.0,
            penalty_coefficient=1000.0,
        ),
    )
    return MonolithicGradientDamageProblem(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=["HEX20", "HEX20"],
        dirichlet_bc_info=[conditions, None],
        gauss_order=[3, 3],
        additional_info=(parameters,),
    )


def write_progress(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def run(arguments):
    conditions = boundary_conditions(0.0)
    problem = create_problem(arguments, conditions)
    options = {
        "petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"},
        "tol": 1e-7,
        "rel_tol": 1e-7,
        "line_search_flag": True,
    }
    records = []

    def solve_step(load, previous_solution):
        conditions[-1][3] = moving_x(load)
        problem.fes[0].update_Dirichlet_boundary_conditions(conditions)
        step_options = dict(options)
        if previous_solution is not None:
            step_options["initial_guess"] = previous_solution
        return solver(problem, solver_options=step_options)

    def accept_step(load, solution):
        problem.update_internal_vars(solution)
        stress = problem.cauchy_stress_quadrature(solution)
        records.append(
            {
                "displacement": load,
                "max_von_mises": float(jnp.max(von_mises_stress(stress))),
                "max_kappa": float(jnp.max(problem.internal_vars[0])),
                "max_damage": float(jnp.max(problem.damage())),
            }
        )
        write_progress(arguments.output, records)

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
    parser.add_argument(
        "--mesh",
        choices=("coarse", "medium", "fine"),
        default="coarse",
    )
    parser.add_argument("--maximum-displacement", type=float, default=25.0)
    parser.add_argument("--young-modulus", type=float, default=210000.0)
    parser.add_argument("--initial-increment", type=float, default=0.25)
    parser.add_argument("--minimum-increment", type=float, default=1e-4)
    parser.add_argument("--maximum-increment", type=float, default=1.0)
    parser.add_argument("--max-runtime", type=float, default=285.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/notched_plate_closed_form/progress.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
