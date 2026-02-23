"""
Tensile test: 10×10×1 mm plate, stretch 1.5 over 51 steps.
Three mesh densities × local and full damage → 6 result sets.
Outputs: same VTK fields as main.py.

Local formulation (ex1_local style): c_d=0 (no gradient), gamma_d active in
damage_model when local_only=True; kappa from SED history.
"""
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_project_root = Path(__file__).resolve().parents[2]
_plate_model = _project_root / "plate_model"
_mrc_code = _project_root / "MRC Code"
sys.path.insert(0, str(_plate_model))
sys.path.insert(0, str(_mrc_code))

from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, Mesh
from damage_model import CoupledDamageProblem

jax.config.update("jax_enable_x64", True)

DAMAGE_THRESHOLD = 0.995
OUT_DIR = Path(__file__).parent / "tensile_test_results"
DOMAIN_X = 10.0
DOMAIN_Y = 10.0
DOMAIN_Z = 1.0
MAX_DISPLACEMENT = 5.0
NUM_LOADS = 1001

E = 210000.0
NU = 0.3
ETA = 0.002
KAPPA_THRESH = 0.1
C_D = 1.0
C_D_LOCAL = 0.0
BETA_D = 1000.0
# Local runs: c_d=C_D_LOCAL (0), gamma_d=1 in damage_model when local_only=True

MESH_CONFIGS = [
    {"name": "non_uniform", "type": "file", "path": "non_uniform_mesh.inp"},
    {"name": "coarse", "type": "box", "nx": 5, "ny": 5, "nz": 1},
    {"name": "medium", "type": "box", "nx": 10, "ny": 10, "nz": 1},
    {"name": "fine", "type": "box", "nx": 20, "ny": 20, "nz": 1},
]


def zero_bc(point):
    return 0.0


def moving_bc(u):
    return lambda point: u


def left_face(point):
    """Whole yz plane at x=0: u_x = 0."""
    return jnp.isclose(point[0], 0.0, atol=1e-5)


def right(point):
    return jnp.isclose(point[0], DOMAIN_X, atol=1e-5)


def corner_origin(point):
    """Single fixed corner (0, 0, 0): u_x = u_y = u_z = 0."""
    return (
        jnp.isclose(point[0], 0.0, atol=1e-5)
        & jnp.isclose(point[1], 0.0, atol=1e-5)
        & jnp.isclose(point[2], 0.0, atol=1e-5)
    )


def point_001(point):
    """Point (0, 0, 1): fix u_x = u_y = 0 to prevent rigid body rotation."""
    return (
        jnp.isclose(point[0], 0.0, atol=1e-5)
        & jnp.isclose(point[1], 0.0, atol=1e-5)
        & jnp.isclose(point[2], DOMAIN_Z, atol=1e-5)
    )


def create_mesh(config):
    if config["type"] == "file":
        import meshio
        mesh_path = os.path.join(os.path.dirname(__file__), config["path"])
        mesh_io = meshio.read(mesh_path)
        return Mesh(mesh_io.points, mesh_io.cells_dict["hexahedron"], ele_type="HEX8")
    else:
        mesh_io = box_mesh(
            Nx=config["nx"], Ny=config["ny"], Nz=config["nz"],
            domain_x=DOMAIN_X, domain_y=DOMAIN_Y, domain_z=DOMAIN_Z,
        )
        return Mesh(mesh_io.points, mesh_io.cells_dict["hexahedron"], ele_type="HEX8")


def get_dirichlet_bc_info(displacement):
    """Corner (0,0,0) fixed; point (0,0,1): u_x=u_y=0; left face (x=0): u_x=0; right face: u_x = displacement."""
    return [
        [corner_origin, corner_origin, corner_origin, point_001, point_001, left_face, right],
        [0, 1, 2, 0, 1, 0, 0],
        [zero_bc, zero_bc, zero_bc, zero_bc, zero_bc, zero_bc, moving_bc(displacement)],
    ]


def _save_step_vtk(coupled_prob, u_sol, phi_sol, step_idx, out_dir, file_prefix):
    kappa = coupled_prob.internal_vars[0]
    damage = 1 - coupled_prob.damage_function(kappa)
    von_mises_quads = coupled_prob.compute_von_mises_quads(u_sol, phi_sol)
    von_mises = jnp.mean(von_mises_quads, axis=1)

    vtk_filename = os.path.join(out_dir, f"{file_prefix}_sol_{step_idx}.vtk")
    save_sol(
        coupled_prob.fe_u, u_sol, vtk_filename,
        point_infos=[("nonlocal_damage", phi_sol)],
        cell_infos=[
            ("kappa", jnp.mean(kappa, axis=1)),
            ("d", jnp.mean(damage, axis=1)),
            ("von_mises", von_mises),
        ],
    )
    print(f"    u: [{float(jnp.min(u_sol)):.6f}, {float(jnp.max(u_sol)):.6f}]")
    print(f"    phi: [{float(jnp.min(phi_sol)):.6f}, {float(jnp.max(phi_sol)):.6f}]")
    print(f"    kappa: [{float(jnp.min(kappa)):.6f}, {float(jnp.max(kappa)):.6f}]")
    print(f"    damage: [{float(jnp.min(damage)):.6f}, {float(jnp.max(damage)):.6f}]")
    print(f"    von_mises: [{float(jnp.min(von_mises)):.6f}, {float(jnp.max(von_mises)):.6f}]")


def run_simulation_nonlocal(mesh, out_dir, file_prefix, local_only=False):
    u_dirichlet_bc_info = get_dirichlet_bc_info(0.0)
    c_d = C_D_LOCAL if local_only else C_D
    additional_info = (E, NU, KAPPA_THRESH, ETA, c_d, BETA_D, local_only)

    coupled_prob = CoupledDamageProblem(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=["HEX8", "HEX8"],
        dirichlet_bc_info=[u_dirichlet_bc_info, None],
        gauss_order=[3, 3],
        additional_info=additional_info,
    )

    solver_options: dict[str, object] = {
        "petsc_solver": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    }

    load_increments = jnp.linspace(0.0, MAX_DISPLACEMENT, NUM_LOADS)
    sol_list = None

    for idx, inc in enumerate(load_increments):
        inc_val = float(inc)
        print(f"  Step {idx + 1}/{NUM_LOADS}: displacement = {inc_val:.6f}")
        u_dirichlet_bc_info[-1][6] = moving_bc(inc_val)
        coupled_prob.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)

        step_options = dict(solver_options)
        if sol_list is not None:
            step_options["initial_guess"] = sol_list

        try:
            u_sol, phi_sol = solver(coupled_prob, solver_options=step_options)
        except (AssertionError, RuntimeError) as e:
            print(f"  Solver diverged at step {idx + 1} (displacement = {inc_val:.6f}): {e}")
            print(f"  Stopping simulation for {file_prefix} due to convergence failure.")
            break
        sol_list = [u_sol, phi_sol]
        coupled_prob.update_internal_vars(sol_list)

        kappa = coupled_prob.internal_vars[0]
        damage = 1 - coupled_prob.damage_function(kappa)
        max_damage = float(jnp.max(damage))
        _save_step_vtk(coupled_prob, u_sol, phi_sol, idx, out_dir, file_prefix)

        if max_damage >= DAMAGE_THRESHOLD:
            print(f"  Stopping: max damage {max_damage:.6f} >= {DAMAGE_THRESHOLD}")
            break


def run_simulation(mesh, mesh_name, local_only, out_dir):
    file_prefix = f"{mesh_name}_{'local' if local_only else 'full'}"
    run_simulation_nonlocal(mesh, out_dir, file_prefix, local_only)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_increments = jnp.linspace(0.0, MAX_DISPLACEMENT, NUM_LOADS)
    print(
        f"Tensile test: 10×10×1 mm, stretch 1.5 over {NUM_LOADS} steps "
        f"(displacement 0 → {MAX_DISPLACEMENT} mm)"
    )
    print(f"Load steps: count={len(load_increments)}")

    for config in MESH_CONFIGS:
        mesh_name = config["name"]
        print(f"\n{'='*70}")
        if config["type"] == "file":
            print(f"Mesh: {mesh_name} (from {config['path']})")
        else:
            print(f"Mesh: {mesh_name} ({config['nx']}×{config['ny']}×{config['nz']})")
        print(f"{'='*70}")
        mesh = create_mesh(config)
        print(f"  Nodes: {mesh.points.shape[0]}, elements: {mesh.cells.shape[0]}")

        for local_only, damage_label in [(True, "local"), (False, "full")]:
            print(f"\n--- {mesh_name} {damage_label} damage ---")
            run_simulation(mesh, mesh_name, local_only, OUT_DIR)

    print(f"\n{'='*70}")
    print(f"Done. Results in {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
