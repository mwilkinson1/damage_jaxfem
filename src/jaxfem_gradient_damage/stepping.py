from dataclasses import dataclass
import signal
from time import monotonic
from typing import Callable, Generic, Optional, TypeVar


Solution = TypeVar("Solution")


def _raise_runtime_limit(_signal_number, _frame):
    raise TimeoutError("simulation runtime limit reached")


@dataclass(frozen=True)
class StepControl:
    initial_increment: float
    minimum_increment: float
    maximum_increment: float
    growth_factor: float = 1.25
    cutback_factor: float = 0.5
    maximum_runtime_seconds: float = 285.0

    def __post_init__(self) -> None:
        if self.minimum_increment <= 0.0:
            raise ValueError("minimum_increment must be positive")
        if not (
            self.minimum_increment
            <= self.initial_increment
            <= self.maximum_increment
        ):
            raise ValueError("initial_increment must lie within step bounds")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must exceed one")
        if not 0.0 < self.cutback_factor < 1.0:
            raise ValueError("cutback_factor must lie between zero and one")
        if not 0.0 < self.maximum_runtime_seconds < 300.0:
            raise ValueError("maximum_runtime_seconds must be below 300")


@dataclass(frozen=True)
class StepResult(Generic[Solution]):
    load: float
    increment: float
    solution: Optional[Solution]
    stop_reason: str
    accepted_steps: int
    rejected_steps: int


def run_adaptive_steps(
    final_load: float,
    control: StepControl,
    solve_step: Callable[[float, Optional[Solution]], Solution],
    accept_step: Callable[[float, Solution], None],
    initial_solution: Optional[Solution] = None,
):
    start_time = monotonic()
    load = 0.0
    increment = control.initial_increment
    solution = initial_solution
    accepted_steps = 0
    rejected_steps = 0

    while load < final_load:
        if monotonic() - start_time >= control.maximum_runtime_seconds:
            return StepResult(
                load,
                increment,
                solution,
                "runtime_limit",
                accepted_steps,
                rejected_steps,
            )

        target_load = min(load + increment, final_load)
        remaining_runtime = max(
            control.maximum_runtime_seconds - (monotonic() - start_time),
            0.001,
        )
        previous_handler = signal.signal(
            signal.SIGALRM,
            _raise_runtime_limit,
        )
        signal.setitimer(signal.ITIMER_REAL, remaining_runtime)
        try:
            candidate = solve_step(target_load, solution)
        except TimeoutError:
            return StepResult(
                load,
                increment,
                solution,
                "runtime_limit",
                accepted_steps,
                rejected_steps,
            )
        except (AssertionError, FloatingPointError, RuntimeError, ValueError):
            rejected_steps += 1
            increment *= control.cutback_factor
            if increment < control.minimum_increment:
                return StepResult(
                    load,
                    increment,
                    solution,
                    "minimum_increment",
                    accepted_steps,
                    rejected_steps,
                )
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)

        solution = candidate
        load = target_load
        accepted_steps += 1
        accept_step(load, solution)
        increment = min(
            increment * control.growth_factor,
            control.maximum_increment,
        )

    return StepResult(
        load,
        increment,
        solution,
        "completed",
        accepted_steps,
        rejected_steps,
    )
