from __future__ import annotations

import copy
import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from graphix.clifford import Clifford
from graphix.random_objects import rand_circuit, rand_state_vector
from graphix.sim.base_backend import NodeIndex
from graphix.sim.statevec import Statevector as SVGraphix
from graphix.sim.statevec import StatevectorBackend as SBGraphix
from graphix.states import BasicStates
from numpy.random import Generator

from graphix_symbolic import Statevector, StatevectorBackend

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from graphix.sim.base_backend import DenseState
    from graphix.states import State
    from numpy.random import PCG64

    _ENCODING = Literal["LSB", "MSB"]


def generate_rnd_data(rng: Generator, nqubits: int) -> npt.NDArray[np.complex128]:
    length = 1 << nqubits
    data = rng.random(length) + 1j * rng.random(length)
    data /= np.sqrt(np.sum(np.abs(data) ** 2))
    return data


class TestStatevector:
    N_JUMPS = 3

    @pytest.mark.parametrize(
        ("state", "data_ref"),
        [
            (BasicStates.PLUS, np.array([1, 1] / np.sqrt(2))),
            (BasicStates.MINUS, np.array([1, -1] / np.sqrt(2))),
            (BasicStates.ZERO, np.array([1, 0])),
            (BasicStates.ONE, np.array([0, 1])),
            (BasicStates.PLUS_I, np.array([1, 1j] / np.sqrt(2))),
            (BasicStates.MINUS_I, np.array([1, -1j] / np.sqrt(2))),
        ],
    )
    def test_init_basic_states(self, state: State, data_ref: npt.NDArray[np.complex128]) -> None:
        sv = Statevector(data=state)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_random_state(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv = Statevector(data)
        assert np.allclose(sv.flatten(), data)

    @pytest.mark.parametrize(
        ("sv", "edge", "data_ref"),
        [
            (Statevector(data=BasicStates.ZERO, nqubit=2), (0, 1), np.array([1, 0, 0, 0])),
            (Statevector(data=[BasicStates.PLUS, BasicStates.PLUS]), (0, 1), np.array([1, 1, 1, -1]) / 2),
            (Statevector(data=[BasicStates.ONE, BasicStates.MINUS]), (0, 1), np.array([0, 0, 1, 1]) / np.sqrt(2)),
            (
                Statevector(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                (0, 2),
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
        ],
    )
    def test_entangle(self, sv: Statevector, edge: tuple[int, int], data_ref: npt.NDArray[np.complex128]) -> None:
        sv.entangle(edge)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "op", "data_ref"),
        [
            (Statevector(data=BasicStates.ZERO, nqubit=2), 0, Clifford.X.matrix, np.array([0, 0, 1, 0])),
            (
                Statevector(data=[BasicStates.PLUS, BasicStates.PLUS]),
                1,
                Clifford.H.matrix,
                np.array([1, 0, 1, 0]) / np.sqrt(2),
            ),
            (
                Statevector(data=[BasicStates.PLUS, BasicStates.MINUS]),
                0,
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                np.array([1, -1, np.exp(0.25j * np.pi), -np.exp(0.25j * np.pi)]) / 2,
            ),
            (
                Statevector(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                1,
                Clifford.Z.matrix,
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
        ],
    )
    def test_evolve_single(
        self, sv: Statevector, q: int, op: npt.NDArray[np.complex128], data_ref: npt.NDArray[np.complex128]
    ) -> None:
        sv.evolve_single(op, q)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "op", "exp_ref"),
        [
            (Statevector(data=BasicStates.ZERO, nqubit=2), 0, Clifford.X.matrix, 0),
            (Statevector(data=[BasicStates.PLUS, BasicStates.PLUS]), 1, Clifford.H.matrix, 1 / np.sqrt(2)),
            (
                Statevector(data=[BasicStates.PLUS, BasicStates.MINUS]),
                0,
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                (1 + np.exp(0.25j * np.pi)) / 2,
            ),
            (
                Statevector(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                1,
                Clifford.Z.matrix,
                0,
            ),
        ],
    )
    def test_expectation_single(
        self, sv: Statevector, q: int, op: npt.NDArray[np.complex128], exp_ref: np.complex128
    ) -> None:
        assert np.isclose(sv.expectation_single(op, q), exp_ref)

    def test_add_nodes(self, fx_rng: Generator) -> None:
        max_qubits = 5
        sv_test = Statevector(nqubit=0)
        psi_ref = np.array([1.0 + 0.0j])

        for _ in range(max_qubits):  # Add a node at each iteration
            data = generate_rnd_data(fx_rng, nqubits=1)
            psi_ref = np.kron(psi_ref, data)
            sv_test.add_nodes(1, data)
            assert np.allclose(sv_test.flatten(), psi_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "sv_ref"),
        [
            (Statevector(data=BasicStates.ZERO, nqubit=2), 0, Statevector(data=BasicStates.ZERO, nqubit=1)),
            (Statevector(data=[BasicStates.PLUS, BasicStates.PLUS]), 1, Statevector(data=BasicStates.PLUS, nqubit=1)),
            (Statevector(data=[BasicStates.PLUS, BasicStates.MINUS]), 0, Statevector(data=BasicStates.MINUS, nqubit=1)),
            (Statevector(data=[BasicStates.ZERO, BasicStates.ONE]), 0, Statevector(data=BasicStates.ONE, nqubit=1)),
            # In previous testcase, branch 1 is 0 (psi_10 == psi_11 == 0), and first element of branch 0 is 0 too (psi_00 == 0)!
            (
                Statevector(data=[BasicStates.PLUS_I, BasicStates.ONE, BasicStates.PLUS]),
                1,
                Statevector(data=[BasicStates.PLUS_I, BasicStates.PLUS], nqubit=2),
            ),
        ],
    )
    def test_remove_qubit(self, sv: Statevector, q: int, sv_ref: Statevector) -> None:
        sv.remove_qubit(q)
        assert np.allclose(sv.flatten(), sv_ref.flatten())

    @pytest.mark.parametrize("permutation", tuple(itertools.permutations(range(3))))
    def test_permute(self, fx_rng: Generator, permutation: Sequence[int]) -> None:
        nqubits = len(permutation)
        statevec = Statevector(rand_state_vector(nqubits, fx_rng))
        statevec_ref = copy.copy(statevec)
        statevec.permute(permutation)
        permute_with_swap(statevec_ref, permutation)
        assert np.array_equal(statevec.psi, statevec_ref.psi)

    def test_permute_bad_permutation(self) -> None:
        statevec = Statevector(nqubit=2)
        with pytest.raises(ValueError, match="Permutation has length"):
            statevec.permute([0])
        with pytest.raises(ValueError, match="not a permutation"):
            statevec.permute([1, 2])


def permute_with_swap(dense_state: DenseState, permutation: Sequence[int]) -> None:
    nqubits = len(permutation)
    node_index = NodeIndex()
    node_index.extend(range(nqubits))
    for i, ind in enumerate(permutation):
        if node_index.index(ind) != i:
            move_from = node_index.index(ind)
            dense_state.swap((i, move_from))
            node_index.swap(i, move_from)


class TestStatevectorGraphix:
    """Tests in this class compare the result against the existing statevector simulator in Graphix. They are not self-contained."""

    N_JUMPS = 3

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_entangle(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        sv_test = Statevector(generate_rnd_data(rng, nqubits))
        sv_ref = SVGraphix(data=sv_test.flatten())
        edge: tuple[int, int] = tuple(rng.choice(range(nqubits), size=2, replace=False))
        for sv in [sv_test, sv_ref]:
            sv.entangle(edge)

        assert sv_ref.isclose(SVGraphix(data=sv_test.flatten()))

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_swap(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        sv_test = Statevector(generate_rnd_data(rng, nqubits))
        sv_ref = SVGraphix(data=sv_test.flatten())
        edge: tuple[int, int] = tuple(rng.choice(range(nqubits), size=2, replace=False))
        for sv in [sv_test, sv_ref]:
            sv.swap(edge)

        assert sv_ref.isclose(SVGraphix(data=sv_test.flatten()))

    def test_evolve_single(self, fx_rng: Generator) -> None:
        nqubits = 5
        for clifford in Clifford:
            sv_test = Statevector(generate_rnd_data(fx_rng, nqubits))
            sv_ref = SVGraphix(data=sv_test.flatten())
            qubit = int(fx_rng.integers(0, nqubits))
            for sv in [sv_test, sv_ref]:
                sv.evolve_single(clifford.matrix, qubit)
            assert sv_ref.isclose(SVGraphix(data=sv_test.flatten()))

    def test_expectation_single(self, fx_rng: Generator) -> None:
        nqubits = 5
        for clifford in Clifford:
            sv_test = Statevector(generate_rnd_data(fx_rng, nqubits))
            sv_ref = SVGraphix(data=sv_test.flatten())
            qubit = int(fx_rng.integers(0, nqubits))

            val_test = sv_test.expectation_single(clifford.matrix, qubit)
            val_ref = sv_ref.expectation_single(clifford.matrix, qubit)

            assert math.isclose(val_test.real, val_ref.real, abs_tol=1e-12)
            assert math.isclose(val_test.imag, val_ref.imag, abs_tol=1e-12)

    def test_add_nodes(self, fx_rng: Generator) -> None:

        max_qubits = 5
        sv_test = Statevector(nqubit=0)
        sv_ref = SVGraphix(nqubit=0)

        for _ in range(max_qubits):  # Add a node at each iteration
            data = generate_rnd_data(fx_rng, nqubits=1)
            sv_test.add_nodes(1, data)
            sv_ref.add_nodes(1, data)

            assert sv_ref.isclose(SVGraphix(data=sv_test.flatten()))

    @pytest.mark.parametrize(
        "projector", [np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)]
    )
    def test_remove_nodes(self, fx_rng: Generator, projector: npt.NDArray[np.complex128]) -> None:

        nqubits = 5
        sv_test = Statevector(generate_rnd_data(fx_rng, nqubits))
        sv_ref = SVGraphix(data=sv_test.flatten())
        q = 0
        for _ in range(nqubits - 1):  # Remove a node at each iteration
            sv_test.evolve_single(projector, q)
            sv_test.remove_qubit(q)
            sv_ref.project_qubit(projector, q)

            assert sv_ref.isclose(SVGraphix(data=sv_test.flatten()))


@pytest.mark.parametrize("jumps", range(1, 6))
def test_pattern_simulator(fx_bg: PCG64, jumps: int) -> None:
    rng = Generator(fx_bg.jumped(jumps))

    nqubits = 5

    pattern = rand_circuit(nqubits, depth=5, rng=rng).transpile().pattern
    pattern.infer_pauli_measurements().remove_pauli_measurements()

    sv_test = pattern.simulate(backend=StatevectorBackend(), rng=rng)
    sv_ref = pattern.simulate(backend=SBGraphix(), rng=rng)

    assert sv_ref.isclose(SVGraphix(data=sv_test.flatten()))
