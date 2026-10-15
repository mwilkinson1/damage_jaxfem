import jax
import jax.numpy as jnp

from jaxfem_gradient_damage.parameters import (
    DamageParameters,
    ElasticParameters,
)


MINIMUM_JACOBIAN = 1e-12


def smooth_positive_part(value, epsilon):
    return 0.5 * (
        value + jnp.sqrt(value * value + epsilon * epsilon)
    )


def damage_degradation(kappa, parameters: DamageParameters):
    excess_history = smooth_positive_part(
        kappa - parameters.threshold,
        parameters.smoothing_epsilon,
    )
    return jnp.exp(-parameters.saturation * excess_history)


def damage_value(kappa, parameters: DamageParameters):
    return 1.0 - damage_degradation(kappa, parameters)


def positive_jacobian(deformation_gradient):
    jacobian = jnp.linalg.det(deformation_gradient)
    return jnp.where(jacobian > MINIMUM_JACOBIAN, jacobian, jnp.nan)


def neo_hookean_energy(
    deformation_gradient,
    parameters: ElasticParameters,
):
    jacobian = positive_jacobian(deformation_gradient)
    first_invariant = jnp.sum(deformation_gradient * deformation_gradient)
    log_jacobian = jnp.log(jacobian)
    return (
        0.5 * parameters.shear_modulus * (first_invariant - 3.0)
        - parameters.shear_modulus * log_jacobian
        + 0.5 * parameters.lame_parameter * log_jacobian**2
    )


def gradient_energy(
    deformation_gradient,
    nonlocal_gradient,
    parameters: DamageParameters,
):
    spatial_gradient = jnp.linalg.solve(
        deformation_gradient.T,
        nonlocal_gradient,
    )
    return (
        0.5
        * parameters.gradient_coefficient
        * jnp.dot(spatial_gradient, spatial_gradient)
    )


def penalty_energy(nonlocal_history, local_history, parameters):
    history_difference = nonlocal_history - local_history
    return 0.5 * parameters.penalty_coefficient * history_difference**2


def internal_energy(
    deformation_gradient,
    nonlocal_history,
    nonlocal_gradient,
    local_history,
    elastic_parameters,
    damage_parameters,
):
    elastic_energy = neo_hookean_energy(
        deformation_gradient,
        elastic_parameters,
    )
    degradation = damage_degradation(local_history, damage_parameters)
    return (
        degradation * elastic_energy
        + gradient_energy(
            deformation_gradient,
            nonlocal_gradient,
            damage_parameters,
        )
        + penalty_energy(
            nonlocal_history,
            local_history,
            damage_parameters,
        )
    )


def degraded_elastic_energy(
    deformation_gradient,
    local_history,
    elastic_parameters,
    damage_parameters,
):
    return damage_degradation(
        local_history,
        damage_parameters,
    ) * neo_hookean_energy(deformation_gradient, elastic_parameters)


degraded_first_piola_stress = jax.grad(
    degraded_elastic_energy,
    argnums=0,
)
first_piola_stress = jax.grad(internal_energy, argnums=0)
nonlocal_flux = jax.grad(gradient_energy, argnums=1)
penalty_force = jax.grad(penalty_energy, argnums=0)


def cauchy_stress(first_piola, deformation_gradient):
    jacobian = positive_jacobian(deformation_gradient)
    return first_piola @ deformation_gradient.T / jacobian


def von_mises_stress(cauchy):
    dimension = cauchy.shape[-1]
    trace = jnp.trace(cauchy, axis1=-2, axis2=-1)
    identity = jnp.eye(dimension, dtype=cauchy.dtype)
    deviatoric = cauchy - trace[..., None, None] * identity / dimension
    return jnp.sqrt(
        1.5 * jnp.sum(deviatoric * deviatoric, axis=(-2, -1))
    )
