from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from jaxfem_gradient_damage.kkt import (
    local_damage_residual,
    update_local_history,
    update_local_history_only,
)
from jaxfem_gradient_damage.parameters import (
    DamageParameters,
    LocalSolverParameters,
)


DAMAGE = DamageParameters(
    threshold=0.2,
    saturation=1.0,
    gradient_coefficient=1.0,
    penalty_coefficient=10.0,
)
SOLVER = LocalSolverParameters(tolerance=1e-11, max_iterations=80)


def update(elastic_energy, nonlocal_history, previous_history):
    return update_local_history(
        elastic_energy,
        nonlocal_history,
        previous_history,
        DAMAGE,
        SOLVER,
    )


def test_inactive_update_preserves_history():
    previous_history = 0.5
    nonlocal_history = 0.5
    updated = update(0.1, nonlocal_history, previous_history)

    np.testing.assert_allclose(updated, previous_history)


def test_active_update_is_irreversible_and_satisfies_residual():
    previous_history = 0.1
    updated = update(1.0, 1.0, previous_history)
    residual = local_damage_residual(
        updated,
        1.0,
        1.0,
        DAMAGE,
    )

    assert np.isfinite(float(updated))
    assert float(updated) >= previous_history
    assert abs(float(residual)) <= 1e-9


def test_local_only_update_is_history_maximum():
    assert update_local_history_only(0.8, 0.3) == 0.8
    assert update_local_history_only(0.2, 0.3) == 0.3


def test_native_state_derivative_matches_finite_difference():
    point = 1.0
    automatic = jax.grad(lambda energy: update(energy, 1.0, 0.1))(point)
    step = 1e-5
    finite_difference = (
        update(point + step, 1.0, 0.1)
        - update(point - step, 1.0, 0.1)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        automatic,
        finite_difference,
        rtol=1e-5,
        atol=1e-7,
    )


def test_native_parameter_derivative_matches_finite_difference():
    def solve_with_saturation(saturation):
        parameters = replace(DAMAGE, saturation=saturation)
        return update_local_history(
            1.0,
            1.0,
            0.1,
            parameters,
            SOLVER,
        )

    automatic = jax.grad(solve_with_saturation)(DAMAGE.saturation)
    step = 1e-5
    finite_difference = (
        solve_with_saturation(DAMAGE.saturation + step)
        - solve_with_saturation(DAMAGE.saturation - step)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        automatic,
        finite_difference,
        rtol=1e-5,
        atol=1e-7,
    )
