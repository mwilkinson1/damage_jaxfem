import jax.numpy as jnp
import numpy as np
import pytest

from jaxfem_gradient_damage.constitutive import damage_value
from jaxfem_gradient_damage.parameters import DamageParameters


@pytest.mark.regression
def test_larger_saturation_increases_damage():
    history = jnp.array(1.0)
    damage = [
        float(
            damage_value(
                history,
                DamageParameters(threshold=0.2, saturation=saturation),
            )
        )
        for saturation in (1.0, 2.0, 5.0, 20.0, 100.0)
    ]

    assert np.all(np.diff(damage) > 0.0)


@pytest.mark.regression
def test_larger_threshold_delays_damage():
    history = jnp.array(1.0)
    damage = [
        float(
            damage_value(
                history,
                DamageParameters(threshold=threshold, saturation=2.0),
            )
        )
        for threshold in (0.0, 0.2, 0.6, 1.0, 2.0)
    ]

    assert np.all(np.diff(damage) <= 0.0)
