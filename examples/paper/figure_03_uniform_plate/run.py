import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from jax_fem.generate_mesh import Mesh, box_mesh
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from jaxfem_gradient_damage import (
    DamageParameters,
    ElasticParameters,
    LocalDamageProblem,
    ModelParameters,
    MonolithicGradientDamageProblem,
    StaggeredGradientDamageProblem,
)
from jaxfem_gradient_damage.constitutive import von_mises_stress
from jaxfem_gradient_damage.stepping import (
    StepControl,
    run_adaptive_steps,
)


jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class SimulationContext:
    arguments: argparse.Namespace
    problem: object
    boundary_conditions: list
    solver_options: dict
    records: list


def zero(_point):
    return 0.0


def moving_x(value):
    return lambda _point: value


def create_boundary_conditions(displacement):
    origin = lambda point: (
        jnp.isclose(point[0], 0.0)
        & jnp.isclose(point[1], 0.0)
        & jnp.isclose(point[2], 0.0)
    )
    edge = lambda point: (
        jnp.isclose(point[0], 0.0)
        & jnp.isclose(point[1], 0.0)
        & jnp.isclose(point[2], 1.0)
    )
    left = lambda point: jnp.isclose(point[0], 0.0)
    right = lambda point: jnp.isclose(point[0], 10.0)
    return [
        [origin, origin, origin, edge, edge, left, right],
        [0, 1, 2, 0, 1, 0, 0],
        [zero, zero, zero, zero, zero, zero, moving_x(displacement)],
    ]


def create_mesh(mesh_name):
    if mesh_name == "non_uniform":
        mesh_data = meshio.read(
            ROOT / "meshes" / "uniform_plate_nonuniform.inp"
        )
        return Mesh(
            mesh_data.points,
            mesh_data.cells_dict["hexahedron"],
            ele_type="HEX8",
        )
    divisions = {"coarse": 5, "medium": 10, "fine": 20}[mesh_name]
    mesh_data = box_mesh(
        Nx=divisions,
        Ny=divisions,
        Nz=1,
        domain_x=10.0,
        domain_y=10.0,
        domain_z=1.0,
    )
    return Mesh(
        mesh_data.points,
        mesh_data.cells_dict["hexahedron"],
        ele_type="HEX8",
    )


def create_problem(arguments, boundary_conditions):
    element_mesh = create_mesh(arguments.mesh)
    gradient_coefficient = (
        0.0 if arguments.formulation == "local" else 1.0
    )
    parameters = ModelParameters(
        elastic=ElasticParameters(arguments.young_modulus, 0.3),
        damage=DamageParameters(
            threshold=0.1,
            saturation=0.002,
            gradient_coefficient=gradient_coefficient,
            penalty_coefficient=1000.0,
        ),
    )
    if arguments.formulation == "local":
        return LocalDamageProblem(
            element_mesh,
            vec=3,
            dim=3,
            ele_type="HEX8",
            dirichlet_bc_info=boundary_conditions,
            gauss_order=3,
            additional_info=(parameters,),
        )
    problem_type = (
        MonolithicGradientDamageProblem
        if arguments.formulation == "monolithic"
        else StaggeredGradientDamageProblem
    )
    return problem_type(
        [element_mesh, element_mesh],
        vec=[3, 1],
        dim=3,
        ele_type=["HEX8", "HEX8"],
        dirichlet_bc_info=[boundary_conditions, None],
        gauss_order=[3, 3],
        additional_info=(parameters,),
    )


def solve_staggered(context, previous_solution):
    problem = context.problem
    previous_load_history = np.asarray(problem.internal_vars[0])
    iterated_history = previous_load_history.copy()
    solution = previous_solution
    for _iteration in range(context.arguments.maximum_staggered_iterations):
        problem.internal_vars = [jnp.asarray(iterated_history)]
        options = dict(context.solver_options)
        if solution is not None:
            options["initial_guess"] = solution
        solution = solver(problem, solver_options=options)
        candidate = np.asarray(
            problem.history_candidate(
                solution,
                jnp.asarray(previous_load_history),
            )
        )
        if np.linalg.norm(candidate - iterated_history) <= 1e-8:
            problem.internal_vars = [jnp.asarray(candidate)]
            return solution
        iterated_history = candidate
    problem.internal_vars = [jnp.asarray(previous_load_history)]
    raise RuntimeError("staggered iteration did not converge")


def solve_step(context, load, previous_solution):
    context.boundary_conditions[-1][6] = moving_x(load)
    context.problem.fes[0].update_Dirichlet_boundary_conditions(
        context.boundary_conditions
    )
    if context.arguments.formulation == "staggered":
        return solve_staggered(context, previous_solution)
    options = dict(context.solver_options)
    if previous_solution is not None:
        options["initial_guess"] = previous_solution
    return solver(context.problem, solver_options=options)


def write_progress(context):
    context.arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with context.arguments.output.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=context.records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(context.records)


def accept_step(context, load, solution):
    problem = context.problem
    if context.arguments.formulation != "staggered":
        problem.update_internal_vars(solution)
    stress_input = (
        solution[0]
        if context.arguments.formulation == "local"
        else solution
    )
    cauchy = problem.cauchy_stress_quadrature(stress_input)
    von_mises = von_mises_stress(cauchy)
    damage = problem.damage()
    context.records.append(
        {
            "displacement": load,
            "max_sigma_11": float(jnp.max(cauchy[..., 0, 0])),
            "max_von_mises": float(jnp.max(von_mises)),
            "max_kappa": float(jnp.max(problem.internal_vars[0])),
            "max_damage": float(jnp.max(damage)),
        }
    )
    write_progress(context)
    if (
        context.arguments.export_final
        and np.isclose(load, context.arguments.maximum_displacement)
    ):
        export_fields(context, load, solution, von_mises)


def export_fields(context, load, solution, von_mises):
    output_path = context.arguments.output.with_suffix(".vtk")
    problem = context.problem
    point_fields = []
    if context.arguments.formulation != "local":
        point_fields.append(("phi", solution[1]))
    save_sol(
        problem.fes[0],
        solution[0],
        str(output_path),
        point_infos=point_fields,
        cell_infos=[
            ("kappa", jnp.mean(problem.internal_vars[0], axis=1)),
            ("d", jnp.mean(problem.damage(), axis=1)),
            ("von_mises", jnp.mean(von_mises, axis=1)),
        ],
    )
    manifest = {
        "displacement": load,
        "mesh": context.arguments.mesh,
        "formulation": context.arguments.formulation,
        "vtk": str(output_path),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2)
    )


def run(arguments):
    boundary_conditions = create_boundary_conditions(0.0)
    context = SimulationContext(
        arguments=arguments,
        problem=create_problem(arguments, boundary_conditions),
        boundary_conditions=boundary_conditions,
        solver_options={
            "petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"},
            "tol": 1e-7,
            "rel_tol": 1e-7,
            "line_search_flag": True,
        },
        records=[],
    )
    control = StepControl(
        initial_increment=arguments.initial_increment,
        minimum_increment=arguments.minimum_increment,
        maximum_increment=arguments.maximum_increment,
        maximum_runtime_seconds=arguments.max_runtime,
    )
    result = run_adaptive_steps(
        arguments.maximum_displacement,
        control,
        lambda load, solution: solve_step(context, load, solution),
        lambda load, solution: accept_step(context, load, solution),
    )
    print(
        f"stop_reason={result.stop_reason}, load={result.load:.6f}, "
        f"accepted={result.accepted_steps}, rejected={result.rejected_steps}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mesh",
        choices=("coarse", "medium", "fine", "non_uniform"),
        default="coarse",
    )
    parser.add_argument(
        "--formulation",
        choices=("local", "monolithic", "staggered"),
        default="monolithic",
    )
    parser.add_argument("--maximum-displacement", type=float, default=5.0)
    parser.add_argument("--young-modulus", type=float, default=210000.0)
    parser.add_argument("--initial-increment", type=float, default=0.005)
    parser.add_argument("--minimum-increment", type=float, default=1e-5)
    parser.add_argument("--maximum-increment", type=float, default=0.01)
    parser.add_argument("--max-runtime", type=float, default=285.0)
    parser.add_argument("--maximum-staggered-iterations", type=int, default=50)
    parser.add_argument("--export-final", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figure_03/progress.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_arguments())
