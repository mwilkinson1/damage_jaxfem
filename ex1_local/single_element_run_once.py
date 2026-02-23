"""
Single run of the single-element uniaxial extension simulation (local-only damage).
Edit the CONFIG block below and run:  python single_element_run_once.py

Local formulation: c_d=0, gamma_d active in damage_model when local_only=True;
kappa from SED history via KKT update (no gradient regularization).
"""
import os
import sys
from pathlib import Path
import csv

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

# ---------------------------------------------------------------------------
# CONFIG: set parameters for this run (local-only)
# ---------------------------------------------------------------------------
E = 210000.0
NU = 0.3
KAPPA_THRESH = 0.1
ETA = 0.002
BETA_D = 1000
C_D_LOCAL = 0.0  # local: no gradient term

# E = 42.0
# NU = 0.45
# KAPPA_THRESH = 0.2
# ETA = 1.0
# BETA_D = 1000

DOMAIN_SIZE = 1.0
MAX_DISPLACEMENT = 0.5
NUM_LOADS = 101
DAMAGE_STOP_THRESHOLD = 1.0
LOCAL_ONLY = True

OUT_DIR = Path(__file__).parent / "single_element_results"
OUTPUT_CSV = None  # None => auto name: single_run_local_eta_{eta}_kappa_{kappa_thresh}.csv

# VTK: save solution every VTK_EVERY steps (and on last step). 0 = no VTK.
VTK_EVERY = 1
# ---------------------------------------------------------------------------


def moving_bc(u):
    return lambda point: u


def create_single_hex_element():
    mesh_io = box_mesh(Nx=1, Ny=1, Nz=1, domain_x=DOMAIN_SIZE, domain_y=DOMAIN_SIZE, domain_z=DOMAIN_SIZE)
    mesh = Mesh(mesh_io.points, mesh_io.cells_dict["hexahedron"], ele_type="HEX8")
    return mesh


def setup_boundary_conditions():
    def left(point):
        return jnp.isclose(point[0], 0.0, atol=1e-5)

    def right(point):
        return jnp.isclose(point[0], DOMAIN_SIZE, atol=1e-5)

    def bottom(point):
        return jnp.isclose(point[1], 0.0, atol=1e-5)

    def front(point):
        return jnp.isclose(point[2], 0.0, atol=1e-5)

    def zero_bc(point):
        return 0.0

    u_dirichlet_bc_info = [
        [left, right, bottom, front],
        [0, 0, 1, 2],
        [zero_bc, moving_bc(0.0), zero_bc, zero_bc],
    ]
    return u_dirichlet_bc_info


def run_simulation():
    mesh = create_single_hex_element()
    u_dirichlet_bc_info = setup_boundary_conditions()

    coupled_prob = CoupledDamageProblem(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=["HEX8", "HEX8"],
        dirichlet_bc_info=[u_dirichlet_bc_info, None],
        gauss_order=[3, 3],
        additional_info=(E, NU, KAPPA_THRESH, ETA, C_D_LOCAL, BETA_D, LOCAL_ONLY),
    )

    solver_options = {
        "petsc_solver": {"ksp_type": "preonly", "pc_type": "lu"},
    }

    load_increments = jnp.linspace(0.0, MAX_DISPLACEMENT, NUM_LOADS)
    increments = []
    displacements = []
    phi_means = []
    kappa_means = []
    damage_means = []
    von_mises_means = []
    sed_means = []

    vtk_dir = OUT_DIR / "vtk"
    if VTK_EVERY > 0:
        vtk_dir.mkdir(parents=True, exist_ok=True)

    sol_list = None
    for idx, inc in enumerate(load_increments):
        print(f"  Increment {idx}/{NUM_LOADS - 1}: disp = {inc:.6f}")

        u_dirichlet_bc_info[-1][1] = moving_bc(inc)
        coupled_prob.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)

        step_options = dict(solver_options)
        if sol_list is not None:
            step_options["initial_guess"] = sol_list

        try:
            u_sol, phi_sol = solver(coupled_prob, solver_options=step_options)
        except (AssertionError, FloatingPointError) as e:
            print(f"  Solver failed at increment {idx}: {e}")
            break

        u_grad = coupled_prob.fe_u.sol_to_grad(u_sol)
        F = u_grad + jnp.eye(coupled_prob.dim)
        sed_quads = jax.vmap(jax.vmap(coupled_prob.psi_neohookean))(F)
        sed_mean_new = float(jnp.mean(sed_quads))

        sol_list = [u_sol, phi_sol]
        coupled_prob.update_internal_vars(sol_list)
        kappa = coupled_prob.internal_vars[0]
        damage = 1 - coupled_prob.damage_function(kappa)

        phi_quad = jnp.squeeze(coupled_prob.fe_phi.convert_from_dof_to_quad(phi_sol), axis=-1)

        sigma = coupled_prob.compute_cauchy_quads(u_sol, phi_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(coupled_prob.dim)
        s_dev = sigma - hydrostatic
        von_mises_quad = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))
        von_mises_cell = jnp.mean(von_mises_quad, axis=1)

        damage_mean = float(jnp.mean(damage))
        increments.append(idx)
        displacements.append(float(inc))
        phi_means.append(float(jnp.mean(phi_quad)))
        kappa_means.append(float(jnp.mean(kappa)))
        damage_means.append(damage_mean)
        von_mises_means.append(float(jnp.mean(von_mises_quad)))
        sed_means.append(sed_mean_new)

        if VTK_EVERY > 0 and (idx % VTK_EVERY == 0 or idx == len(load_increments) - 1):
            vtk_path = vtk_dir / f"single_run_local_sol_{idx:04d}.vtk"
            save_sol(
                coupled_prob.fe_u, u_sol, str(vtk_path),
                point_infos=[("phi", phi_sol)],
                cell_infos=[
                    ("kappa", jnp.mean(kappa, axis=1)),
                    ("d", jnp.mean(damage, axis=1)),
                    ("von_mises", von_mises_cell),
                    ("sed", jnp.mean(sed_quads, axis=1)),
                ],
            )
            print(f"    VTK: {vtk_path.name}")

        if damage_mean >= DAMAGE_STOP_THRESHOLD:
            break

    return increments, displacements, phi_means, kappa_means, damage_means, von_mises_means, sed_means


def save_to_csv(filename):
    increments, displacements, phi_means, kappa_means, damage_means, von_mises_means, sed_means = run_simulation()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "increment", "prescribed_displacement",
                "phi_mean", "kappa_mean", "damage_mean", "von_mises_mean", "sed_mean",
            ],
        )
        writer.writeheader()
        for i, disp, phi, kappa, dmg, vm, sed in zip(
            increments, displacements, phi_means, kappa_means,
            damage_means, von_mises_means, sed_means,
        ):
            writer.writerow({
                "increment": i,
                "prescribed_displacement": disp,
                "phi_mean": phi,
                "kappa_mean": kappa,
                "damage_mean": dmg,
                "von_mises_mean": vm,
                "sed_mean": sed,
            })
    print(f"Results saved to {path}")
    return path


def main():
    filename = OUTPUT_CSV
    if filename is None:
        filename = f"single_run_local_eta_{ETA}_kappa_{KAPPA_THRESH}.csv"
    print(f"Running single simulation (local-only): eta={ETA}, kappa_thresh={KAPPA_THRESH}, num_loads={NUM_LOADS}")
    print("=" * 60)
    save_to_csv(filename)


if __name__ == "__main__":
    main()
