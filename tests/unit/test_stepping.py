from jaxfem_gradient_damage.stepping import (
    StepControl,
    run_adaptive_steps,
)


def test_successful_steps_reach_final_load():
    accepted_loads = []
    control = StepControl(
        initial_increment=0.1,
        minimum_increment=0.01,
        maximum_increment=0.25,
        maximum_runtime_seconds=10.0,
    )

    result = run_adaptive_steps(
        0.5,
        control,
        lambda load, _solution: load,
        lambda load, _solution: accepted_loads.append(load),
    )

    assert result.stop_reason == "completed"
    assert result.load == 0.5
    assert accepted_loads[-1] == 0.5


def test_failed_step_is_cut_back():
    attempted_loads = []
    control = StepControl(
        initial_increment=0.2,
        minimum_increment=0.025,
        maximum_increment=0.2,
        maximum_runtime_seconds=10.0,
    )

    def solve(load, _solution):
        attempted_loads.append(load)
        if len(attempted_loads) == 1:
            raise RuntimeError("intentional failure")
        return load

    result = run_adaptive_steps(
        0.2,
        control,
        solve,
        lambda _load, _solution: None,
    )

    assert result.stop_reason == "completed"
    assert result.rejected_steps == 1
    assert attempted_loads[:2] == [0.2, 0.1]
