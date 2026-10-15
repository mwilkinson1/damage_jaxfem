import jax.numpy as jnp
import numpy as np

from jaxfem_gradient_damage.constitutive import von_mises_stress


def test_hydrostatic_stress_has_zero_von_mises():
    stress = 3.2 * jnp.eye(3)

    np.testing.assert_allclose(von_mises_stress(stress), 0.0, atol=1e-12)


def test_uniaxial_von_mises_equals_axial_stress_magnitude():
    stress = jnp.diag(jnp.array([5.0, 0.0, 0.0]))

    np.testing.assert_allclose(von_mises_stress(stress), 5.0, atol=1e-12)
