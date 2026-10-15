from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ElasticParameters:
    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        if self.young_modulus <= 0.0:
            raise ValueError("young_modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie between -1 and 0.5")

    @property
    def shear_modulus(self) -> float:
        return self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def lame_parameter(self) -> float:
        numerator = self.young_modulus * self.poisson_ratio
        denominator = (
            (1.0 + self.poisson_ratio)
            * (1.0 - 2.0 * self.poisson_ratio)
        )
        return numerator / denominator

    def tree_flatten(self):
        return (self.young_modulus, self.poisson_ratio), None

    @classmethod
    def tree_unflatten(cls, _auxiliary_data, children):
        instance = object.__new__(cls)
        object.__setattr__(instance, "young_modulus", children[0])
        object.__setattr__(instance, "poisson_ratio", children[1])
        return instance


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DamageParameters:
    threshold: float
    saturation: float
    gradient_coefficient: float = 0.0
    penalty_coefficient: float = 0.0
    smoothing_epsilon: float = 1e-8
    degradation_floor: float = 1e-13

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ValueError("threshold must be nonnegative")
        if self.saturation <= 0.0:
            raise ValueError("saturation must be positive")
        if self.gradient_coefficient < 0.0:
            raise ValueError("gradient_coefficient must be nonnegative")
        if self.penalty_coefficient < 0.0:
            raise ValueError("penalty_coefficient must be nonnegative")
        if self.smoothing_epsilon <= 0.0:
            raise ValueError("smoothing_epsilon must be positive")
        if not 0.0 < self.degradation_floor < 1.0:
            raise ValueError("degradation_floor must lie between zero and one")

    def tree_flatten(self):
        children = (
            self.threshold,
            self.saturation,
            self.gradient_coefficient,
            self.penalty_coefficient,
            self.smoothing_epsilon,
            self.degradation_floor,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, _auxiliary_data, children):
        instance = object.__new__(cls)
        attribute_names = (
            "threshold",
            "saturation",
            "gradient_coefficient",
            "penalty_coefficient",
            "smoothing_epsilon",
            "degradation_floor",
        )
        for attribute_name, value in zip(attribute_names, children):
            object.__setattr__(instance, attribute_name, value)
        return instance


@dataclass(frozen=True)
class LocalSolverParameters:
    tolerance: float = 1e-10
    max_bracket_iterations: int = 50
    max_iterations: int = 100
    tangent_floor: float = 1e-12

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        if self.max_bracket_iterations < 1:
            raise ValueError("max_bracket_iterations must be positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tangent_floor <= 0.0:
            raise ValueError("tangent_floor must be positive")


@dataclass(frozen=True)
class ModelParameters:
    elastic: ElasticParameters
    damage: DamageParameters
    local_solver: LocalSolverParameters = LocalSolverParameters()
