from functools import partial

import jax
import jax.numpy as jnp

from jaxfem_gradient_damage.constitutive import damage_degradation
from jaxfem_gradient_damage.parameters import (
    DamageParameters,
    LocalSolverParameters,
)


def local_damage_residual(
    local_history,
    elastic_energy,
    nonlocal_history,
    damage_parameters,
    coupling_scale=1.0,
):
    degradation = damage_degradation(local_history, damage_parameters)
    safe_degradation = jnp.maximum(
        degradation,
        damage_parameters.degradation_floor,
    )
    penalty_drive = (
        coupling_scale
        * damage_parameters.penalty_coefficient
        * (nonlocal_history - local_history)
        / (damage_parameters.saturation * safe_degradation)
    )
    return elastic_energy + penalty_drive - local_history


def _expand_upper_bound(residual, lower, initial_upper, iterations):
    def expand(_, state):
        upper, upper_residual = state
        needs_expansion = upper_residual > 0.0
        candidate = 2.0 * upper
        candidate_residual = residual(candidate)
        return (
            jnp.where(needs_expansion, candidate, upper),
            jnp.where(needs_expansion, candidate_residual, upper_residual),
        )

    initial_residual = residual(initial_upper)
    upper, upper_residual = jax.lax.fori_loop(
        0,
        iterations,
        expand,
        (initial_upper, initial_residual),
    )
    return lower, upper, upper_residual


def _solve_bracketed_root(residual, lower, upper, parameters):
    derivative = jax.grad(residual)

    def solve(_, state):
        bracket_lower, bracket_upper, estimate = state
        estimate_residual = residual(estimate)
        estimate_derivative = derivative(estimate)
        derivative_is_safe = (
            jnp.abs(estimate_derivative) > parameters.tangent_floor
        )
        safe_derivative = jnp.where(
            derivative_is_safe,
            estimate_derivative,
            1.0,
        )
        newton_estimate = estimate - estimate_residual / safe_derivative
        midpoint = 0.5 * (bracket_lower + bracket_upper)
        use_newton = (
            derivative_is_safe
            & (newton_estimate > bracket_lower)
            & (newton_estimate < bracket_upper)
        )
        candidate = jnp.where(use_newton, newton_estimate, midpoint)
        candidate_residual = residual(candidate)
        new_lower = jnp.where(
            candidate_residual > 0.0,
            candidate,
            bracket_lower,
        )
        new_upper = jnp.where(
            candidate_residual > 0.0,
            bracket_upper,
            candidate,
        )
        converged = (
            jnp.abs(estimate_residual) <= parameters.tolerance
        )
        return jax.lax.cond(
            converged,
            lambda: state,
            lambda: (new_lower, new_upper, candidate),
        )

    initial_estimate = 0.5 * (lower + upper)
    _, _, root = jax.lax.fori_loop(
        0,
        parameters.max_iterations,
        solve,
        (lower, upper, initial_estimate),
    )
    return root


@partial(jax.jit, static_argnames=("solver_parameters",))
def update_local_history(
    elastic_energy,
    nonlocal_history,
    previous_history,
    damage_parameters,
    solver_parameters=LocalSolverParameters(),
    coupling_scale=1.0,
):
    def residual_increment(history_increment):
        local_history = previous_history + history_increment
        return local_damage_residual(
            local_history,
            elastic_energy,
            nonlocal_history,
            damage_parameters,
            coupling_scale,
        )

    zero_increment = jnp.zeros_like(elastic_energy)
    trial_residual = residual_increment(zero_increment)
    initial_upper = jnp.maximum(
        jnp.maximum(elastic_energy - previous_history, 0.0),
        1.0,
    )
    lower, upper, upper_residual = _expand_upper_bound(
        residual_increment,
        zero_increment,
        initial_upper,
        solver_parameters.max_bracket_iterations,
    )
    root_increment = _solve_bracketed_root(
        residual_increment,
        lower,
        upper,
        solver_parameters,
    )
    valid_active_root = (
        (trial_residual > 0.0)
        & (upper_residual <= 0.0)
        & (
            jnp.abs(residual_increment(root_increment))
            <= 10.0 * solver_parameters.tolerance
        )
    )
    active_history = previous_history + jnp.maximum(root_increment, 0.0)
    failed_active_history = jnp.full_like(previous_history, jnp.nan)
    return jnp.where(
        trial_residual <= 0.0,
        previous_history,
        jnp.where(valid_active_root, active_history, failed_active_history),
    )


def update_local_history_only(elastic_energy, previous_history):
    return jnp.maximum(previous_history, elastic_energy)


def local_update_diagnostics(
    local_history,
    elastic_energy,
    nonlocal_history,
    previous_history,
    damage_parameters,
    coupling_scale=1.0,
):
    residual = local_damage_residual(
        local_history,
        elastic_energy,
        nonlocal_history,
        damage_parameters,
        coupling_scale,
    )
    return {
        "residual": residual,
        "is_finite": jnp.isfinite(local_history),
        "is_irreversible": local_history >= previous_history,
    }
