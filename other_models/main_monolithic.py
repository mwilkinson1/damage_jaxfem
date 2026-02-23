import logging
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh
from damage_model_monolithic import CoupledDamageProblemMonolithic

PROFILE_JAX_FEM = True
jax_fem_log_level = logging.DEBUG if PROFILE_JAX_FEM else logging.ERROR
logging.getLogger("jax_fem").setLevel(jax_fem_log_level)

jax.config.update("jax_enable_x64", True)

MESH_FILE = Path(__file__).with_name("notched_plate_C3D20_coarse.inp")
OUT_DIR = Path("monolithic_damage_results")
E = 210000.0
NU = 0.3
ETA = 0.002
KAPPA_THRESH = 0.1
C_D = 1.0
BETA_D = 100.0
LOAD_MAX = 25.0
NUM_LOADS = 251
VTK_EVERY = 1
# For stronger coupling or damage, increase NUM_LOADS (e.g. 501) for smaller steps
INIT_PHI_KAPPA = 1e-6

SOLVER = "petsc"


def left(point):
    return jnp.isclose(point[0], -100.0, atol=1e-5)


def right(point):
    return jnp.isclose(point[0], 100.0, atol=1e-5)


def zero_bc(point):
    return 0.0


def moving_bc(u):
    return lambda point: u


def main():
    mesh_path = Path(MESH_FILE).expanduser().resolve()
    out_dir = OUT_DIR / mesh_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_import = meshio.read(mesh_path)
    mesh = Mesh(mesh_import.points, mesh_import.cells_dict["hexahedron20"])
    ele_type = "HEX20"

    u_dirichlet_bc_info = [
        [left] * 3 + [right],
        [0, 1, 2, 0],
        [zero_bc] * 3 + [moving_bc(0.0)],
    ]

    problem = CoupledDamageProblemMonolithic(
        [mesh, mesh, mesh],
        vec=[3, 1, 1],
        dim=3,
        ele_type=[ele_type, ele_type, ele_type],
        dirichlet_bc_info=[u_dirichlet_bc_info, None, None],
        gauss_order=[3, 3, 3],
        additional_info=(E, NU, KAPPA_THRESH, ETA, C_D, BETA_D),
    )

    n_u = problem.fe_u.num_total_nodes
    n_phi = problem.fe_phi.num_total_nodes
    n_kappa = problem.fe_kappa.num_total_nodes
    initial_guess = [
        jnp.zeros((n_u, 3)),
        jnp.full((n_phi, 1), INIT_PHI_KAPPA),
        jnp.full((n_kappa, 1), INIT_PHI_KAPPA),
    ]

    common = {"tol": 1e-5, "rel_tol": 1e-6, 
    # "line_search_flag": True
    }
    if SOLVER == "umfpack":
        solver_options = {"umfpack_solver": {}, **common}
    elif SOLVER == "petsc_iterative":
        solver_options = {
            "petsc_solver": {"ksp_type": "gmres", "pc_type": "ilu"},
            **common,
        }
    else:
        solver_options = {
            "petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"},
            **common,
        }

    load_increments = jnp.linspace(0.0, LOAD_MAX, NUM_LOADS)
    step_sizes = np.diff(np.array(load_increments))
    print(
        f"Load steps: count={len(load_increments)}, min_step={step_sizes.min():.6f}, "
        f"max_step={step_sizes.max():.6f}"
    )

    sol_list = None
    for idx, inc in enumerate(load_increments):
        if idx == 0:
            step_size = float(inc)
        else:
            step_size = float(inc - load_increments[idx - 1])

        print("=" * 100)
        print(f"Loading increment: {inc},   step size: {step_size:.6f}")
        print("=" * 100)
        u_dirichlet_bc_info[-1][3] = moving_bc(inc)
        problem.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)

        step_options = dict(solver_options)
        step_options["initial_guess"] = sol_list if sol_list is not None else initial_guess

        sol_list = solver(problem, solver_options=step_options)
        u_sol, phi_sol, kappa_sol = sol_list[0], sol_list[1], sol_list[2]

        save_this_step = (idx % VTK_EVERY == 0) or (idx == len(load_increments) - 1)
        if save_this_step:
            kappa_quad = problem.fe_kappa.convert_from_dof_to_quad(kappa_sol)
            kappa_quad = jnp.squeeze(kappa_quad, axis=-1)
            kappa_cell = jnp.mean(kappa_quad, axis=1)
            damage_quad = 1 - problem.damage_function(kappa_quad)
            damage_cell = jnp.mean(damage_quad, axis=1)
            von_mises = problem.compute_von_mises(u_sol, phi_sol, kappa_sol)
            nodal_von_mises = problem.compute_nodal_von_mises(
                u_sol, phi_sol, kappa_sol
            )

            vtk_filename = os.path.join(out_dir, f"local_sol_monolithic_{idx}.vtk")
            save_sol(
                problem.fe_u,
                u_sol,
                vtk_filename,
                point_infos=[
                    ("nonlocal_damage", phi_sol),
                    ("kappa", kappa_sol),
                    ("von_mises", nodal_von_mises),
                ],
                cell_infos=[
                    ("kappa", kappa_cell),
                    ("d", damage_cell),
                    ("von_mises", von_mises),
                ],
            )

            print(f"u: [{float(jnp.min(u_sol)):.6f}, {float(jnp.max(u_sol)):.6f}]")
            print(f"phi: [{float(jnp.min(phi_sol)):.6f}, {float(jnp.max(phi_sol)):.6f}]")
            print(
                f"kappa: [{float(jnp.min(kappa_quad)):.6f}, {float(jnp.max(kappa_quad)):.6f}]"
            )
            print(
                f"damage: [{float(jnp.min(damage_quad)):.6f}, {float(jnp.max(damage_quad)):.6f}]"
            )
            print(
                f"max_stress: [{float(jnp.min(nodal_von_mises)):.6f}, "
                f"{float(jnp.max(nodal_von_mises)):.6f}]"
            )


if __name__ == "__main__":
    main()
