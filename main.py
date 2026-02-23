import csv
import os
import sys
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import meshio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import Mesh
from damage_model import CoupledDamageProblem, get_boundary_force_component

# Constants (float64 for numerical precision)
MESH_FILE = Path(__file__).with_name("notched_plate_C3D20_fine.inp")
OUT_DIR = Path("coupled_damage_results")
E = 210000.0
NU = 0.3
ETA = 0.002
KAPPA_THRESH = 0.1
C_D = 1.0
BETA_D = 1000.0
LOAD_MAX = 25.0
NUM_LOADS = 51

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

    coupled_prob = CoupledDamageProblem(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        dirichlet_bc_info=[u_dirichlet_bc_info, None],
        gauss_order=[3, 3],
        additional_info=(E, NU, KAPPA_THRESH, ETA, C_D, BETA_D, False)
    )

    solver_options = {
        'petsc_solver': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        },
    }

    # Linear load schedule
    load_increments = jnp.linspace(0.0, LOAD_MAX, NUM_LOADS)
    step_sizes = np.diff(np.array(load_increments))
    print(f"Load steps: count={len(load_increments)}, min_step={step_sizes.min():.6f}, max_step={step_sizes.max():.6f}")

    sol_list = None
    force_records = []
    force_csv_path = out_dir / "fine_force.csv"

    for idx, inc in enumerate(load_increments):
        if idx == 0:
            step_size = float(inc)
        else:
            step_size = float(inc - load_increments[idx - 1])

        print(f"Loading increment: {inc},   step size: {step_size:.6f}")
        u_dirichlet_bc_info[-1][3] = moving_bc(inc)
        coupled_prob.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)

        step_options = dict(solver_options)
        if sol_list is not None:
            step_options["initial_guess"] = sol_list

        u_sol, phi_sol = solver(coupled_prob, solver_options=step_options)
        sol_list = [u_sol, phi_sol]
        coupled_prob.update_internal_vars(sol_list)

        force_x_left = get_boundary_force_component(
            coupled_prob, sol_list, fe_index=0, bc_group_index=0
        )
        force_records.append({"step": idx, "load": float(inc), "force_x_left": force_x_left})

        kappa = coupled_prob.internal_vars[0]
        damage = 1 - coupled_prob.damage_function(kappa)
        von_mises_quads = coupled_prob.compute_von_mises_quads(u_sol, phi_sol)
        von_mises = jnp.mean(von_mises_quads, axis=1)

        vtk_filename = os.path.join(out_dir, f"fine_sol_{idx}.vtk")
        save_sol(
            coupled_prob.fe_u, u_sol, vtk_filename,
            point_infos=[('nonlocal_damage', phi_sol)],
            cell_infos=[
                ('kappa', jnp.mean(kappa, axis=1)),
                ('d', jnp.mean(damage, axis=1)),
                ('von_mises', von_mises),
            ],
        )

        with open(force_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "load", "force_x_left"])
            writer.writeheader()
            writer.writerows(force_records)

        print(f"force_x_left: {force_x_left:.6f}")
        print(f"u: [{float(jnp.min(u_sol)):.6f}, {float(jnp.max(u_sol)):.6f}]")
        print(f"phi: [{float(jnp.min(phi_sol)):.6f}, {float(jnp.max(phi_sol)):.6f}]")  
        print(f"kappa: [{float(jnp.min(kappa)):.6f}, {float(jnp.max(kappa)):.6f}]")
        print(f"damage: [{float(jnp.min(damage)):.6f}, {float(jnp.max(damage)):.6f}]")
        print(f"von_mises: [{float(jnp.min(von_mises)):.6f}, {float(jnp.max(von_mises)):.6f}]")


if __name__ == "__main__":
    main()
