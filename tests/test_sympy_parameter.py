from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix import Circuit
from graphix.branch_selector import RandomBranchSelector
from numpy.random import Generator

from graphix_symbolic import DensityMatrixBackend, StatevectorBackend, SympyParameter

if TYPE_CHECKING:
    from graphix.parameter import Parameter
    from graphix.sim.base_backend import DenseStateBackend


def test_parameter_circuit_simulation(fx_rng: Generator) -> None:
    alpha = SympyParameter("alpha")
    circuit = Circuit(1)
    circuit.rz(0, alpha)
    result_subs_then_simulate = circuit.subs(alpha, 0.5).simulate().state
    assert result_subs_then_simulate.psi.dtype == np.complex128
    result_simulate_then_subs = circuit.simulate(
        backend=StatevectorBackend(branch_selector=RandomBranchSelector(pr_calc=False), symbolic=True)
    ).state.subs(alpha, 0.5)
    assert np.allclose(result_subs_then_simulate.flatten(), result_simulate_then_subs.psi)


def test_parameter_parallel_substitution(fx_rng: Generator) -> None:
    alpha = SympyParameter("alpha")
    beta = SympyParameter("beta")
    circuit = Circuit(2)
    circuit.rz(0, alpha)
    circuit.rz(1, beta)
    mapping: dict[Parameter, float] = {alpha: 0.5, beta: 0.4}
    result_subs_then_simulate = circuit.xreplace(mapping).simulate().state
    result_simulate_then_subs = circuit.simulate(
        backend=StatevectorBackend(branch_selector=RandomBranchSelector(pr_calc=False), symbolic=True)
    ).state.xreplace(mapping)
    assert np.allclose(result_subs_then_simulate.flatten(), result_simulate_then_subs.flatten())


@pytest.mark.parametrize("backend", ["statevector", "densitymatrix"])
@pytest.mark.filterwarnings("ignore:Simulating using densitymatrix backend with no noise.")
def test_parameter_pattern_simulation(backend, fx_rng: Generator) -> None:
    alpha = SympyParameter("alpha")
    circuit = Circuit(1)
    circuit.rz(0, alpha)
    pattern = circuit.transpile().pattern
    result_subs_then_simulate = pattern.subs(alpha, 0.5).simulate(backend, rng=fx_rng)
    # We cannot compute probabilities on symbolic states; we explore
    # one arbitrary branch.
    symb_backend: DenseStateBackend
    if backend == "statevector":
        symb_backend = StatevectorBackend(branch_selector=RandomBranchSelector(pr_calc=False), symbolic=True)
    elif backend == "densitymatrix":
        symb_backend = DensityMatrixBackend(branch_selector=RandomBranchSelector(pr_calc=False), symbolic=True)

    result_simulate_then_subs = pattern.simulate(backend=symb_backend, rng=fx_rng).subs(alpha, 0.5)
    if backend == "statevector":
        assert np.allclose(result_subs_then_simulate.flatten(), result_simulate_then_subs.flatten())
    elif backend == "densitymatrix":
        assert np.allclose(result_subs_then_simulate.rho, result_simulate_then_subs.rho)
