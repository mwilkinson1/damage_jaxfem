from jaxfem_gradient_damage.problems.gradient_monolithic import (
    MonolithicGradientDamageProblem,
)
from jaxfem_gradient_damage.problems.gradient_staggered import (
    StaggeredGradientDamageProblem,
)
from jaxfem_gradient_damage.problems.local import LocalDamageProblem

__all__ = [
    "LocalDamageProblem",
    "MonolithicGradientDamageProblem",
    "StaggeredGradientDamageProblem",
]
