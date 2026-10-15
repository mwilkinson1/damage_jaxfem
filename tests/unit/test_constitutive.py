import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxfem_gradient_damage.constitutive import (
    damage_degradation,
    gradient_energy,
    internal_energy,
    neo_hookean_energy,
    positive_jacobian,
)
from jaxfem_gradient_damage.parameters import (
    DamageParameters,
    ElasticParameters,
)


ELASTIC = ElasticParameters(young_modulus=42.0, poisson_ratio=0.45)
DAMAGE = DamageParameters(
    threshold=0.2,
    saturation=2.0,
    gradient_coefficient=1.0,
    penalty_coefficient=1000.0,
)


def test_identity_has_zero_elastic_energy_and_stress():
    identity = jnp.eye(3)
    energy = neo_hookean_energy(identity, ELASTIC)
    stress = jax.grad(neo_hookean_energy)(identity, ELASTIC)

    np.testing.assert_allclose(energy, 0.0, atol=1e-12)
    np.testing.assert_allclose(stress, 0.0, atol=1e-12)


def test_gradient_energy_matches_right_cauchy_green_form():
    deformation_gradient = jnp.array(
        [[1.2, 0.1, 0.0], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]]
    )
    nonlocal_gradient = jnp.array([0.4, -0.2, 0.3])
    right_cauchy_green = (
        deformation_gradient.T @ deformation_gradient
    )
    expected = 0.5 * DAMAGE.gradient_coefficient * (
        nonlocal_gradient
        @ jnp.linalg.inv(right_cauchy_green)
        @ nonlocal_gradient
    )

    actual = gradient_energy(
        deformation_gradient,
        nonlocal_gradient,
        DAMAGE,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_degradation_is_bounded_and_monotone():
    history = jnp.linspace(0.0, 5.0, 101)
    degradation = damage_degradation(history, DAMAGE)

    assert np.all(np.asarray(degradation) > 0.0)
    assert np.all(np.asarray(degradation) <= 1.0)
    assert np.all(np.diff(np.asarray(degradation)) <= 1e-14)


def test_nonpositive_jacobian_is_rejected():
    inverted = jnp.diag(jnp.array([-1.0, 1.0, 1.0]))

    assert np.isnan(float(positive_jacobian(inverted)))
    assert np.isnan(float(neo_hookean_energy(inverted, ELASTIC)))


def test_internal_energy_directional_derivative():
    deformation_gradient = jnp.array(
        [[1.1, 0.03, 0.0], [0.0, 0.95, 0.02], [0.0, 0.0, 1.04]]
    )
    direction = jnp.array(
        [[0.2, -0.1, 0.0], [0.05, 0.1, 0.0], [0.0, 0.03, -0.1]]
    )
    arguments = (
        0.4,
        jnp.array([0.2, -0.1, 0.05]),
        0.35,
        ELASTIC,
        DAMAGE,
    )
    automatic = jnp.sum(
        jax.grad(internal_energy)(deformation_gradient, *arguments)
        * direction
    )
    step = 1e-6
    forward = internal_energy(
        deformation_gradient + step * direction,
        *arguments,
    )
    backward = internal_energy(
        deformation_gradient - step * direction,
        *arguments,
    )
    finite_difference = (forward - backward) / (2.0 * step)

    np.testing.assert_allclose(
        automatic,
        finite_difference,
        rtol=1e-6,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"young_modulus": 0.0, "poisson_ratio": 0.3},
        {"young_modulus": 1.0, "poisson_ratio": 0.5},
    ],
)
def test_invalid_elastic_parameters_fail(parameters):
    with pytest.raises(ValueError):
        ElasticParameters(**parameters)
