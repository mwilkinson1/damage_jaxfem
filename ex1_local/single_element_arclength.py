"""
Single-element uniaxial extension with arc-length continuation (local-only damage).

Uses the displacement-driven arc-length solver from plate_model/jax_fem/solver.py
to follow the equilibrium path through softening and avoid the discontinuous
stress drop that occurs with fixed displacement increments.
"""

import os
import sys
from pathlib import Path
import csv

import jax
import jax.numpy as jnp
import numpy as np

# Prefer plate_model jax_fem (has arc_length_solver_disp_driven)
_project_root = Path(__file__).resolve().parents[2]
_plate_model = _project_root / "plate_model"
_mrc_code = _project_root / "MRC Code"
sys.path.insert(0, str(_plate_model))
sys.path.insert(0, str(_mrc_code))

from jax_fem.solver import solver, arc_length_solver_disp_driven
from jax_fem.generate_mesh import box_mesh, Mesh
from damage_model import CoupledDamageProblem

jax.config.update("jax_enable_x64", True)

# Material parameters (match single_element.py)
E = 42.0
NU = 0.45
BETA_D = 1e-6
C_D = 1e-6

ETA_VALS = [1.0, 2.0, 5.0, 20.0, 100.0]
KAPPA_VALS = [0.0, 0.2, 0.6, 1.0, 2.0]
KAPPA_THRESH_ETA = 0.2
ETA_FIXED = 2.0

DOMAIN_SIZE = 1.0
MAX_DISPLACEMENT = 0.5
OUT_DIR = Path(__file__).parent / "single_element_results"
LOCAL_ONLY = True

# Arc-length parameters (smaller Delta_l and larger psi discourage branch switching)
DELTA_L = 0.01
PSI = 2.0
MAX_ARC_STEPS = 1000


def reference_bc_one(_point):
    """Reference displacement = 1.0 for arc-length (actual displacement = lambda)."""
    return 1.0


def create_single_hex_element():
    mesh_io = box_mesh(
        Nx=1, Ny=1, Nz=1,
        domain_x=DOMAIN_SIZE, domain_y=DOMAIN_SIZE, domain_z=DOMAIN_SIZE,
    )
    points = mesh_io.points
    cells = mesh_io.cells_dict["hexahedron"]
    return Mesh(points, cells, ele_type="HEX8")


def setup_boundary_conditions_reference():
    """BC with reference value 1.0 on right face so lambda = displacement."""
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
        [zero_bc, reference_bc_one, zero_bc, zero_bc],
    ]
    return u_dirichlet_bc_info


def run_simulation_arclength(kappa_thresh, eta, out_dir):
    print(f"\n{'='*70}")
    print(f"Arc-length (local-only): kappa_thresh={kappa_thresh} MPa, eta={eta} MPa^-1")
    print(f"{'='*70}")

    mesh = create_single_hex_element()
    ele_type = "HEX8"
    u_dirichlet_bc_info = setup_boundary_conditions_reference()

    coupled_prob = CoupledDamageProblem(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        dirichlet_bc_info=[u_dirichlet_bc_info, None],
        gauss_order=[3, 3],
        additional_info=(E, NU, kappa_thresh, eta, C_D, BETA_D, LOCAL_ONLY),
    )

    num_dofs = coupled_prob.num_total_dofs_all_vars
    prev_u_vec = np.zeros(num_dofs)
    prev_lamda = 0.0
    prev_Delta_u_vec = np.zeros(num_dofs)
    prev_Delta_lamda = 0.0

    increments = []
    displacements = []
    phi_means = []
    kappa_means = []
    damage_means = []
    von_mises_means = []

    step = 0
    while step < MAX_ARC_STEPS and prev_lamda < MAX_DISPLACEMENT:
        try:
            u_vec, lamda, Delta_u_vec_dir, Delta_lamda_dir = arc_length_solver_disp_driven(
                coupled_prob,
                prev_u_vec,
                prev_lamda,
                prev_Delta_u_vec,
                prev_Delta_lamda,
                Delta_l=DELTA_L,
                psi=PSI,
            )
        except (AssertionError, FloatingPointError) as e:
            print(f"  Arc-length step {step} failed: {e}")
            break

        u_vec = np.array(u_vec)
        lamda = float(lamda)
        sol_list = coupled_prob.unflatten_fn_sol_list(u_vec)
        coupled_prob.update_internal_vars(sol_list)

        u_sol, phi_sol = sol_list
        kappa = coupled_prob.internal_vars[0]
        damage = 1 - coupled_prob.damage_function(kappa)

        phi_quad = coupled_prob.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi_quad = jnp.squeeze(phi_quad, axis=-1)

        sigma = coupled_prob.compute_cauchy_quads(u_sol, phi_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(coupled_prob.dim)
        s_dev = sigma - hydrostatic
        von_mises_quad = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))

        increments.append(step)
        displacements.append(lamda)
        phi_means.append(float(jnp.mean(phi_quad)))
        kappa_means.append(float(jnp.mean(kappa)))
        damage_means.append(float(jnp.mean(damage)))
        von_mises_means.append(float(jnp.mean(von_mises_quad)))

        if step % 20 == 0:
            print(f"  Step {step}: lamda = {lamda:.6f}, von_mises_mean = {von_mises_means[-1]:.6f}")

        prev_u_vec = u_vec
        prev_lamda = lamda
        prev_Delta_u_vec = Delta_u_vec_dir
        prev_Delta_lamda = Delta_lamda_dir
        step += 1

    return increments, displacements, phi_means, kappa_means, damage_means, von_mises_means


def save_to_csv(filename, kappa_thresh, eta, increments, displacements,
                phi_means, kappa_means, damage_means, von_mises_means):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "increment", "prescribed_displacement",
            "phi_mean", "kappa_mean", "damage_mean", "von_mises_mean",
        ])
        writer.writeheader()
        for idx, disp, phi, kappa, dmg, vm in zip(
            increments, displacements, phi_means, kappa_means,
            damage_means, von_mises_means,
        ):
            writer.writerow({
                "increment": idx,
                "prescribed_displacement": disp,
                "phi_mean": phi,
                "kappa_mean": kappa,
                "damage_mean": dmg,
                "von_mises_mean": vm,
            })


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Single hex element with arc-length continuation (local-only)")

    for eta in ETA_VALS:
        inc, disp, phi, kappa, dmg, vm = run_simulation_arclength(
            KAPPA_THRESH_ETA, eta, out_dir
        )
        path = out_dir / f"arclength_eta_{eta:.1f}_kappa_{KAPPA_THRESH_ETA:.1f}.csv"
        save_to_csv(path, KAPPA_THRESH_ETA, eta, inc, disp, phi, kappa, dmg, vm)
        print(f"  Saved: {path}")

    for kappa_th in KAPPA_VALS:
        inc, disp, phi, kappa, dmg, vm = run_simulation_arclength(
            kappa_th, ETA_FIXED, out_dir
        )
        path = out_dir / f"arclength_eta_{ETA_FIXED:.1f}_kappa_{kappa_th:.1f}.csv"
        save_to_csv(path, kappa_th, ETA_FIXED, inc, disp, phi, kappa, dmg, vm)
        print(f"  Saved: {path}")

    print("\nArc-length runs complete. Use plot_single_element_results.py with arclength_*.csv for figures.")


if __name__ == "__main__":
    main()
