import os
import sys
from pathlib import Path
import csv

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_fem.solver import solver
from jax_fem.generate_mesh import box_mesh, Mesh
from damage_model import CoupledDamageProblem

jax.config.update("jax_enable_x64", True)

# Material parameters from material_point_validation.py
E = 42.0          # MPa
NU = 0.45
BETA_D = 1e-6
C_D = 1e-6

# Parameter sweep values
ETA_VALS = [1.0, 2.0, 5.0, 20.0, 100.0]  # MPa^-1
KAPPA_VALS = [0.0, 0.2, 0.6, 1.0, 2.0]   # MPa
KAPPA_THRESH_ETA = 0.2                   # MPa (for eta sweep)
ETA_FIXED = 2.0                          # MPa^-1 (for kappa sweep)

# Uniaxial extension parameters
DOMAIN_SIZE = 1.0
MAX_DISPLACEMENT = 0.5  # lambda = 1.5 at end
NUM_LOADS = 1001  # finer steps for softening regime
OUT_DIR = Path(__file__).parent / "single_element_results"

# Local-only damage: c_d=0, kappa=phi (no gradient regularization)
LOCAL_ONLY = True


def moving_bc(u):
    return lambda point: u


def create_single_hex_element():
    """Create a single 3D hexahedral element."""
    mesh_io = box_mesh(Nx=1, Ny=1, Nz=1, domain_x=DOMAIN_SIZE, domain_y=DOMAIN_SIZE, domain_z=DOMAIN_SIZE)
    points = mesh_io.points
    cells = mesh_io.cells_dict['hexahedron']
    mesh = Mesh(points, cells, ele_type='HEX8')
    return mesh


def setup_boundary_conditions():
    """Set up boundary conditions for uniaxial extension."""
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
        [zero_bc, moving_bc(0.0), zero_bc, zero_bc]
    ]

    return u_dirichlet_bc_info


def run_simulation(kappa_thresh, eta, out_dir):
    """Run uniaxial extension simulation for given parameters (local-only damage)."""
    print(f"\n{'='*70}")
    print(f"Running simulation (local-only): kappa_thresh={kappa_thresh} MPa, eta={eta} MPa^-1")
    print(f"{'='*70}")

    mesh = create_single_hex_element()
    ele_type = 'HEX8'
    u_dirichlet_bc_info = setup_boundary_conditions()

    coupled_prob = CoupledDamageProblem(
        [mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        dirichlet_bc_info=[u_dirichlet_bc_info, None],
        gauss_order=[3, 3],
        additional_info=(E, NU, kappa_thresh, eta, C_D, BETA_D, LOCAL_ONLY)
    )

    solver_options = {
        'petsc_solver': {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
        },
    }

    load_increments = jnp.linspace(0.0, MAX_DISPLACEMENT, NUM_LOADS)
    step_sizes = np.diff(np.array(load_increments))

    increments = []
    displacements = []
    phi_means = []
    kappa_means = []
    damage_means = []
    von_mises_means = []

    sol_list = None
    for idx, inc in enumerate(load_increments):
        if idx % 5 == 0:
            print(f"  Loading increment {idx}/{NUM_LOADS-1}: displacement = {inc:.6f}")

        u_dirichlet_bc_info[-1][1] = moving_bc(inc)
        coupled_prob.fes[0].update_Dirichlet_boundary_conditions(u_dirichlet_bc_info)

        step_options = dict(solver_options)
        if sol_list is not None:
            step_options["initial_guess"] = sol_list

        u_sol, phi_sol = solver(coupled_prob, solver_options=step_options)
        sol_list = [u_sol, phi_sol]
        coupled_prob.update_internal_vars(sol_list)

        kappa = coupled_prob.internal_vars[0]
        damage = 1 - coupled_prob.damage_function(kappa)

        phi_quad = coupled_prob.fe_phi.convert_from_dof_to_quad(phi_sol)
        phi_quad = jnp.squeeze(phi_quad, axis=-1)

        sigma = coupled_prob.compute_cauchy_quads(u_sol, phi_sol)
        trace_sigma = jnp.trace(sigma, axis1=-2, axis2=-1)
        hydrostatic = trace_sigma[..., None, None] / 3.0 * jnp.eye(coupled_prob.dim)
        s_dev = sigma - hydrostatic
        von_mises_quad = jnp.sqrt(3.0 / 2.0 * jnp.sum(s_dev * s_dev, axis=(-2, -1)))

        increments.append(idx)
        displacements.append(float(inc))
        phi_means.append(float(jnp.mean(phi_quad)))
        kappa_means.append(float(jnp.mean(kappa)))
        damage_means.append(float(jnp.mean(damage)))
        von_mises_means.append(float(jnp.mean(von_mises_quad)))

    return increments, displacements, phi_means, kappa_means, damage_means, von_mises_means


def save_to_csv(filename, kappa_thresh, eta, increments, displacements,
                phi_means, kappa_means, damage_means, von_mises_means):
    """Save simulation results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'increment', 'prescribed_displacement',
            'phi_mean', 'kappa_mean', 'damage_mean', 'von_mises_mean'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, disp, phi, kappa, dmg, vm in zip(
            increments, displacements, phi_means, kappa_means,
            damage_means, von_mises_means
        ):
            writer.writerow({
                'increment': idx,
                'prescribed_displacement': disp,
                'phi_mean': phi,
                'kappa_mean': kappa,
                'damage_mean': dmg,
                'von_mises_mean': vm
            })


def main():
    """Run parameter sweeps for uniaxial extension simulation (local-only damage)."""
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Creating single hex element mesh (local-only damage)...")
    mesh = create_single_hex_element()
    print(f"Mesh created: {len(mesh.points)} nodes, {len(mesh.cells)} elements")

    # Sweep 1: vary eta_d with fixed kappa_thresh = 0.2 MPa
    print(f"\n{'='*70}")
    print("SWEEP 1: Varying eta_d with fixed kappa_thresh = 0.2 MPa (local-only)")
    print(f"{'='*70}")

    for eta in ETA_VALS:
        increments, displacements, phi_means, kappa_means, damage_means, von_mises_means = \
            run_simulation(KAPPA_THRESH_ETA, eta, out_dir)

        filename = out_dir / f"eta_{eta:.1f}_kappa_{KAPPA_THRESH_ETA:.1f}.csv"
        save_to_csv(filename, KAPPA_THRESH_ETA, eta, increments, displacements,
                    phi_means, kappa_means, damage_means, von_mises_means)
        print(f"  Saved results to: {filename}")

    # Sweep 2: vary kappa_thresh with fixed eta = 2.0 MPa^-1
    print(f"\n{'='*70}")
    print("SWEEP 2: Varying kappa_thresh with fixed eta = 2.0 MPa^-1 (local-only)")
    print(f"{'='*70}")

    for kappa_th in KAPPA_VALS:
        increments, displacements, phi_means, kappa_means, damage_means, von_mises_means = \
            run_simulation(kappa_th, ETA_FIXED, out_dir)

        filename = out_dir / f"eta_{ETA_FIXED:.1f}_kappa_{kappa_th:.1f}.csv"
        save_to_csv(filename, kappa_th, ETA_FIXED, increments, displacements,
                    phi_means, kappa_means, damage_means, von_mises_means)
        print(f"  Saved results to: {filename}")

    print(f"\n{'='*70}")
    print("All simulations complete (local-only)!")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
