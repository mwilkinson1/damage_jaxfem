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
from damage_model_staggered import CoupledDamageProblemStaggered

# Set to logging.DEBUG to see where time is spent (e.g. "Function split_and_compute_cell took X.XX s", solver steps)
# Set to logging.INFO to see only high-level "Solve took X [s]" per load step
# Set to logging.ERROR to silence jax_fem (default)
PROFILE_JAX_FEM = True  # True => DEBUG (verbose timing), False => ERROR (quiet)

jax_fem_log_level = logging.DEBUG if PROFILE_JAX_FEM else logging.ERROR
logging.getLogger("jax_fem").setLevel(jax_fem_log_level)

jax.config.update("jax_enable_x64", True)

# Constants
MESH_FILE = Path(__file__).with_name("notched_plate_C3D20_coarse.inp")
OUT_DIR = Path("staggered_damage_results")
E = 210000.0
NU = 0.3
ETA = 0.002
KAPPA_THRESH = 0.1
C_D = 1.0
BETA_D = 100.0
LOAD_MAX = 25.0
NUM_LOADS = 251
VTK_EVERY = 1  # Write VTK and compute von Mises only every N steps (and on last step)

# Inner staggered loop: iterate (global solve, local kappa update) until kappa converges within each load step
INNER_STAGGERED = True
KAPPA_TOL = 1e-8
MAX_INNER_ITERS = 50

# Solver: 'petsc' (direct LU), 'umfpack' (SciPy/UMFPACK, often faster), or 'petsc_iterative' (GMRES+ILU, can be faster for large systems)
SOLVER = 'petsc'

# Boundary condition functions
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
    ele_type = 'HEX20'

    u_dirichlet_bc_info = [
        [left]*3 + [right],
        [0, 1, 2, 0],
        [zero_bc]*3 + [moving_bc(0.0)]
    ]

    problem = CoupledDamageProblemStaggered(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        dirichlet_bc_info=[u_dirichlet_bc_info, None],
        gauss_order=[3, 3],
        additional_info=(E, NU, KAPPA_THRESH, ETA, C_D, BETA_D)
    )

    common = {'tol': 1e-5, 'rel_tol': 1e-6}
    if SOLVER == 'umfpack':
        solver_options = {'umfpack_solver': {}, **common}
    elif SOLVER == 'petsc_iterative':
        solver_options = {
            'petsc_solver': {'ksp_type': 'gmres', 'pc_type': 'ilu'},
            **common,
        }
    else:
        solver_options = {
            'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'},
            **common,
        }

    load_increments = jnp.linspace(0.0, LOAD_MAX, NUM_LOADS)
    step_sizes = np.diff(np.array(load_increments))
    print(f"Load steps: count={len(load_increments)}, min_step={step_sizes.min():.6f}, max_step={step_sizes.max():.6f}")

    sol_list = None
    for idx, inc in enumerate(load_increments):
        if idx == 0:
            step_size = float(inc)
        else:
            step_size = float(inc - load_increments[idx - 1])

        print("="*100)
        print(f"Loading increment: {inc},   step size: {step_size:.6f}")
        print("="*100)
        u_dirichlet_bc_info[-1][3] = moving_bc(inc)
        problem.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)

        step_options = dict(solver_options)
        if sol_list is not None:
            step_options["initial_guess"] = sol_list

        if INNER_STAGGERED:
            kappa_n = np.array(problem.internal_vars[0])
            kappa_k = np.array(kappa_n)
            kappa_diff_norm = np.nan
            for inner in range(MAX_INNER_ITERS):
                problem.internal_vars[0] = jnp.array(kappa_k)
                u_sol, phi_sol = solver(problem, solver_options=step_options)
                step_options["initial_guess"] = [u_sol, phi_sol]
                kappa_k_plus_1 = np.array(
                    problem.local_update_kappa_field(u_sol, phi_sol, jnp.array(kappa_n))
                )
                kappa_diff_norm = np.linalg.norm(kappa_k_plus_1 - kappa_k)
                if kappa_diff_norm < KAPPA_TOL:
                    if inner > 0:
                        print(f"  Inner kappa converged in {inner + 1} iterations (||Δκ||={kappa_diff_norm:.2e})")
                    break
                kappa_k = kappa_k_plus_1
            else:
                print(f"  Warning: inner kappa did not converge in {MAX_INNER_ITERS} iters (||Δκ||={kappa_diff_norm:.2e})")
            problem.internal_vars[0] = jnp.array(kappa_k_plus_1)
            sol_list = [u_sol, phi_sol]
        else:
            u_sol, phi_sol = solver(problem, solver_options=step_options)
            sol_list = [u_sol, phi_sol]
            problem.update_internal_vars(sol_list)

        save_this_step = (idx % VTK_EVERY == 0) or (idx == len(load_increments) - 1)
        if save_this_step:
            kappa = problem.internal_vars[0]
            damage = 1 - problem.damage_function(kappa)
            von_mises = problem.compute_von_mises(u_sol, phi_sol)
            nodal_von_mises = problem.compute_nodal_von_mises(u_sol, phi_sol)

            vtk_filename = os.path.join(out_dir, f"local_sol_staggered_{idx}.vtk")
            save_sol(
                problem.fe_u, u_sol, vtk_filename,
                point_infos=[('nonlocal_damage', phi_sol), ('von_mises', nodal_von_mises)],
                cell_infos=[('kappa', jnp.mean(kappa, axis=1)),
                            ('d', jnp.mean(damage, axis=1)),
                            ('von_mises', von_mises)]
            )

            print(f"u: [{float(jnp.min(u_sol)):.6f}, {float(jnp.max(u_sol)):.6f}]")
            print(f"phi: [{float(jnp.min(phi_sol)):.6f}, {float(jnp.max(phi_sol)):.6f}]")
            print(f"kappa: [{float(jnp.min(kappa)):.6f}, {float(jnp.max(kappa)):.6f}]")
            print(f"damage: [{float(jnp.min(damage)):.6f}, {float(jnp.max(damage)):.6f}]")
            print(f"max_stress: [{float(jnp.min(nodal_von_mises)):.6f}, {float(jnp.max(nodal_von_mises)):.6f}]")


if __name__ == "__main__":
    main()
