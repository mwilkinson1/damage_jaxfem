import logging
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from jax_fem.utils import save_sol

from damage_model_decoupled import DisplacementProblemDecoupled, PhaseFieldProblemDecoupled

PROFILE_JAX_FEM = True
jax_fem_log_level = logging.DEBUG if PROFILE_JAX_FEM else logging.ERROR
logging.getLogger("jax_fem").setLevel(jax_fem_log_level)

jax.config.update("jax_enable_x64", True)

MESH_FILE = Path(__file__).with_name("notched_plate_C3D20_coarse.inp")
OUT_DIR = Path("decoupled_damage_results")
E = 210000.0
NU = 0.3
ETA = 0.002
KAPPA_THRESH = 0.1
C_D = 1.0
BETA_D = 100.0
LOAD_MAX = 25.0
NUM_LOADS = 251
VTK_EVERY = 1

KAPPA_TOL = 1e-8
MAX_PHI_ITERS = 500

SOLVER = "petsc"
# Quicker options for the (more linear) phi problem: same backend with lighter settings,
# or e.g. 'petsc_iterative' (GMRES+Jacobi) / 'umfpack'. Looser Newton tol often sufficient.
SOLVER_PHI = "petsc_iterative"
PHI_TOL = 1e-4
PHI_REL_TOL = 1e-5


def left(point):
    return jnp.isclose(point[0], -100.0, atol=1e-5)


def right(point):
    return jnp.isclose(point[0], 100.0, atol=1e-5)


def zero_bc(_point):
    return 0.0


def moving_bc(u_value):
    return lambda _point: u_value


def build_solver_options(solver_name=None, tol=None, rel_tol=None):
    solver_name = solver_name or SOLVER
    common = {
        "tol": tol if tol is not None else 1e-5,
        "rel_tol": rel_tol if rel_tol is not None else 1e-6,
    }
    if solver_name == "umfpack":
        return {"umfpack_solver": {}, **common}
    if solver_name == "petsc_iterative":
        return {
            "petsc_solver": {"ksp_type": "gmres", "pc_type": "ilu"},
            **common,
        }
    if solver_name == "petsc_iterative_jacobi":
        return {
            "petsc_solver": {"ksp_type": "gmres", "pc_type": "jacobi"},
            **common,
        }
    return {
        "petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"},
        **common,
    }


def build_u_solver_options():
    return build_solver_options(SOLVER)


def build_phi_solver_options():
    return build_solver_options(
        solver_name=SOLVER_PHI,
        tol=PHI_TOL,
        rel_tol=PHI_REL_TOL,
    )


def create_u_dirichlet_bc_info(load_value):
    return [[left] * 3 + [right], [0, 1, 2, 0], [zero_bc] * 3 + [moving_bc(load_value)]]


def solve_displacement(u_problem, u_dirichlet_bc_info, load_value, solver_options, u_guess):
    u_dirichlet_bc_info[-1][3] = moving_bc(load_value)
    u_problem.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)
    step_options = dict(solver_options)
    if u_guess is not None:
        step_options["initial_guess"] = [u_guess]
    return solver(u_problem, solver_options=step_options)[0]


def solve_phase_field(phi_problem, sed_field, kappa_n, phi_guess, solver_options):
    kappa_k = np.array(kappa_n)
    kappa_diff_norm = np.nan
    step_options = dict(solver_options)
    if phi_guess is not None:
        step_options["initial_guess"] = [phi_guess]

    for inner in range(MAX_PHI_ITERS):
        phi_problem.set_kappa_field(jnp.array(kappa_k))
        phi_sol = solver(phi_problem, solver_options=step_options)[0]
        step_options["initial_guess"] = [phi_sol]

        kappa_k_plus_1 = np.array(
            phi_problem.local_update_kappa_from_sed(sed_field, phi_sol, jnp.array(kappa_n))
        )
        kappa_diff_norm = np.linalg.norm(kappa_k_plus_1 - kappa_k)
        if kappa_diff_norm < KAPPA_TOL:
            if inner > 0:
                print(f"  Phi/kappa converged in {inner + 1} iterations (||Δκ||={kappa_diff_norm:.2e})")
            return phi_sol, jnp.array(kappa_k_plus_1)
        kappa_k = kappa_k_plus_1

    print(f"  Warning: phi/kappa did not converge in {MAX_PHI_ITERS} iters (||Δκ||={kappa_diff_norm:.2e})")
    return phi_sol, jnp.array(kappa_k_plus_1)


def write_outputs(out_dir, idx, u_problem, u_sol, phi_sol, kappa):
    damage = 1 - u_problem.damage_function(kappa)
    von_mises = u_problem.compute_von_mises(u_sol)
    nodal_von_mises = u_problem.compute_nodal_von_mises(u_sol)

    vtk_filename = os.path.join(out_dir, f"local_sol_decoupled_{idx}.vtk")
    save_sol(
        u_problem.fe_u,
        u_sol,
        vtk_filename,
        point_infos=[("nonlocal_damage", phi_sol), ("von_mises", nodal_von_mises)],
        cell_infos=[
            ("kappa", jnp.mean(kappa, axis=1)),
            ("d", jnp.mean(damage, axis=1)),
            ("von_mises", von_mises),
        ],
    )

    print(f"u: [{float(jnp.min(u_sol)):.6f}, {float(jnp.max(u_sol)):.6f}]")
    print(f"phi: [{float(jnp.min(phi_sol)):.6f}, {float(jnp.max(phi_sol)):.6f}]")
    print(f"kappa: [{float(jnp.min(kappa)):.6f}, {float(jnp.max(kappa)):.6f}]")
    print(f"damage: [{float(jnp.min(damage)):.6f}, {float(jnp.max(damage)):.6f}]")
    print(f"max_stress: [{float(jnp.min(nodal_von_mises)):.6f}, {float(jnp.max(nodal_von_mises)):.6f}]")


def main():
    mesh_path = Path(MESH_FILE).expanduser().resolve()
    out_dir = OUT_DIR / mesh_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_import = meshio.read(mesh_path)
    mesh = Mesh(mesh_import.points, mesh_import.cells_dict["hexahedron20"])
    ele_type = "HEX20"

    u_problem = DisplacementProblemDecoupled(
        mesh,
        vec=3,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=create_u_dirichlet_bc_info(0.0),
        gauss_order=3,
        additional_info=(E, NU, KAPPA_THRESH, ETA, C_D, BETA_D),
    )
    phi_problem = PhaseFieldProblemDecoupled(
        mesh,
        vec=1,
        dim=3,
        ele_type=ele_type,
        dirichlet_bc_info=None,
        gauss_order=3,
        additional_info=(E, NU, KAPPA_THRESH, ETA, C_D, BETA_D),
    )

    u_solver_options = build_u_solver_options()
    phi_solver_options = build_phi_solver_options()
    load_increments = jnp.linspace(0.0, LOAD_MAX, NUM_LOADS)
    step_sizes = np.diff(np.array(load_increments))
    print(
        f"Load steps: count={len(load_increments)}, min_step={step_sizes.min():.6f}, "
        f"max_step={step_sizes.max():.6f}"
    )

    kappa = jnp.zeros((u_problem.fe_u.num_cells, u_problem.fe_u.num_quads))
    u_sol = None
    phi_sol = None
    u_dirichlet_bc_info = create_u_dirichlet_bc_info(0.0)

    for idx, load_value in enumerate(load_increments):
        if idx == 0:
            step_size = float(load_value)
        else:
            step_size = float(load_value - load_increments[idx - 1])

        print("=" * 100)
        print(f"Loading increment: {load_value},   step size: {step_size:.6f}")
        print("=" * 100)

        u_problem.set_kappa_field(kappa)
        u_sol = solve_displacement(
            u_problem=u_problem,
            u_dirichlet_bc_info=u_dirichlet_bc_info,
            load_value=load_value,
            solver_options=u_solver_options,
            u_guess=u_sol,
        )

        sed_field = u_problem.compute_sed_field(u_sol)
        phi_sol, kappa = solve_phase_field(
            phi_problem=phi_problem,
            sed_field=sed_field,
            kappa_n=kappa,
            phi_guess=phi_sol,
            solver_options=phi_solver_options,
        )

        u_problem.set_kappa_field(kappa)
        phi_problem.set_kappa_field(kappa)

        save_this_step = (idx % VTK_EVERY == 0) or (idx == len(load_increments) - 1)
        if save_this_step:
            write_outputs(out_dir, idx, u_problem, u_sol, phi_sol, kappa)


if __name__ == "__main__":
    main()
