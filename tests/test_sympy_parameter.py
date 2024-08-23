import numpy as np
import pytest
from graphix import Circuit
from numpy.random import Generator

from graphix_symbolic import SympyParameter


def test_parameter_circuit_simulation(fx_rng: Generator) -> None:
    alpha = SympyParameter("alpha")
    circuit = Circuit(1)
    circuit.rz(0, alpha)
    result_subs_then_simulate = circuit.subs(alpha, 0.5).simulate_statevector().statevec
    result_simulate_then_subs = circuit.simulate_statevector().statevec.subs(alpha, 0.5)
    assert np.allclose(result_subs_then_simulate.psi, result_simulate_then_subs.psi)


def test_parameter_parallel_substitution(fx_rng: Generator) -> None:
    alpha = SympyParameter("alpha")
    beta = SympyParameter("beta")
    circuit = Circuit(2)
    circuit.rz(0, alpha)
    circuit.rz(1, beta)
    mapping = {alpha: 0.5, beta: 0.4}
    result_subs_then_simulate = circuit.xreplace(mapping).simulate_statevector().statevec
    result_simulate_then_subs = circuit.simulate_statevector().statevec.xreplace(mapping)
    assert np.allclose(result_subs_then_simulate.psi, result_simulate_then_subs.psi)


@pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
def test_parameter_pattern_simulation(backend, fx_rng: Generator) -> None:
    alpha = SympyParameter("alpha")
    circuit = Circuit(1)
    circuit.rz(0, alpha)
    pattern = circuit.transpile().pattern
    # Both simulations (numeric and symbolic) will use the same
    # seed for random number generation, to ensure that the
    # explored branch is the same for the two simulations.
    seed = fx_rng.integers(2**63)
    result_subs_then_simulate = pattern.subs(alpha, 0.5).simulate_pattern(
        backend, pr_calc=False, rng=np.random.default_rng(seed)
    )
    # Note: pr_calc=False is mandatory since we cannot compute
    # probabilities on symbolic states; we explore one arbitrary
    # branch.
    result_simulate_then_subs = pattern.simulate_pattern(backend, pr_calc=False, rng=np.random.default_rng(seed)).subs(
        alpha, 0.5
    )
    if backend == "statevector":
        assert np.allclose(result_subs_then_simulate.psi, result_simulate_then_subs.psi)
    elif backend == "densitymatrix":
        assert np.allclose(result_subs_then_simulate.rho, result_simulate_then_subs.rho)
