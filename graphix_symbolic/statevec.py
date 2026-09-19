"""MBQC state vector backend supporting symbolic computations."""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat

import numpy as np
import numpy.typing as npt
from graphix import states
from graphix.parameter import (
    Expression,
    ExpressionOrSupportsComplex,
    InplaceParameterizable,
    check_expression_or_float,
    with_parameter,
    with_parameters,
)
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix, kron, tensordot
from graphix.sim.statevec import _check_permutation
from graphix.states import BasicStates

# override introduced in Python 3.12
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    from graphix.parameter import ExpressionOrFloat, ExpressionOrSupportsFloat, Parameter
    from graphix.sim.data import Data

    _ENCODING = Literal["LSB", "MSB"]


CZ_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [1, 0]], [[0, 0], [0, -1]]]],
    dtype=np.complex128,
)
CNOT_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [[[0, 0], [0, 1]], [[0, 0], [1, 0]]]],
    dtype=np.complex128,
)
SWAP_TENSOR = np.array(
    [[[[1, 0], [0, 0]], [[0, 0], [1, 0]]], [[[0, 1], [0, 0]], [[0, 0], [0, 1]]]],
    dtype=np.complex128,
)


class Statevector(DenseState, InplaceParameterizable):
    """Statevector object."""

    psi: Matrix

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
    ) -> None:
        """Initialize statevector objects.

        `data` can be:
        - a single :class:`graphix.states.State` (classical description of a quantum state)
        - an iterable of :class:`graphix.states.State` objects
        - an iterable of scalars (A 2**n numerical statevector)
        - a *graphix.statevec.Statevector* object

        If *nqubit* is not provided, the number of qubit is inferred from *data* and checked for consistency.
        If only one :class:`graphix.states.State` is provided and nqubit is a valid integer, initialize the statevector
        in the tensor product state.
        If both *nqubit* and *data* are provided, consistency of the dimensions is checked.
        If a *graphix.statevec.Statevector* is passed, returns a copy.

        Parameters
        ----------
        data : Data, optional
            input data to prepare the state. Can be a classical description or a numerical input, defaults to graphix.states.BasicStates.PLUS
        nqubit : int, optional
            number of qubits to prepare, defaults to None
        """
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        if isinstance(data, Statevector):
            # assert nqubit is None or len(state.flatten()) == 2**nqubit
            if nqubit is not None and len(data.flatten()) != 2**nqubit:
                raise ValueError(
                    f"Inconsistent parameters between nqubit = {nqubit} and the inferred number of qubit = {len(data.flatten())}."
                )
            self.psi = data.psi.copy()
            return

        # The type
        # list[states.State] | list[ExpressionOrSupportsComplex] | list[Iterable[ExpressionOrSupportsComplex]]
        # would be more precise, but given a value X of type Iterable[A] | Iterable[B],
        # mypy infers that list(X) has type list[A | B] instead of list[A] | list[B].
        input_list: list[states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex]]
        if isinstance(data, states.State):
            if nqubit is None:
                nqubit = 1
            input_list = [data] * nqubit
        elif isinstance(data, Iterable):
            input_list = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}")

        if len(input_list) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("nqubit is not null but input state is empty.")

            self.psi = np.array(1, dtype=np.complex128)

        elif isinstance(input_list[0], states.State):
            if nqubit is None:
                nqubit = len(input_list)
            elif nqubit != len(input_list):
                raise ValueError("Mismatch between nqubit and length of input state.")

            def state_to_statevector(
                s: states.State | ExpressionOrSupportsComplex | Iterable[ExpressionOrSupportsComplex],
            ) -> npt.NDArray[np.complex128]:
                if not isinstance(s, states.State):
                    raise TypeError("Data should be an homogeneous sequence of states.")
                return s.to_statevector_numpy()

            list_of_sv = [state_to_statevector(s) for s in input_list]

            tmp_psi = functools.reduce(lambda m0, m1: np.kron(m0, m1).astype(np.complex128), list_of_sv)
            # reshape
            self.psi = tmp_psi.reshape((2,) * nqubit)
        # `SupportsFloat` is needed because `numpy.float64` is not an instance of `SupportsComplex`!
        elif isinstance(input_list[0], (Expression, SupportsComplex, SupportsFloat)):
            if nqubit is None:
                length = len(input_list)
                if length & (length - 1):
                    raise ValueError("Length is not a power of two")
                nqubit = length.bit_length() - 1
            elif nqubit != len(input_list).bit_length() - 1:
                raise ValueError("Mismatch between nqubit and length of input state")
            psi = np.array(input_list)
            # check only if the matrix is not symbolic
            if psi.dtype != "O" and not np.allclose(np.sqrt(np.sum(np.abs(psi) ** 2)), 1):
                raise ValueError("Input state is not normalized")
            self.psi = psi.reshape((2,) * nqubit)
        else:
            raise TypeError(f"First element of data has type {type(input_list[0])} whereas Number or State is expected")

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""
        Add nodes (qubits) to the state vector and initialize them in a specified state.

        Parameters
        ----------
        nqubit : int
            The number of qubits to add to the state vector.

        data : Data, optional
            The state in which to initialize the newly added nodes.

            - If a single basic state is provided, all new nodes are initialized in that state.
            - If a list of basic states is provided, it must match the length of ``nodes``, and
              each node is initialized with its corresponding state.
            - A single-qubit state vector will be broadcast to all nodes.
            - A multi-qubit state vector of dimension :math:`2^n`, where :math:`n = \mathrm{len}(nodes)`,
              initializes the new nodes jointly.

        Notes
        -----
        Previously existing nodes remain unchanged.
        """
        sv_to_add = Statevector(nqubit=nqubit, data=data)
        self.tensor(sv_to_add)

    @override
    def evolve_single(self, op: Matrix, qubit: int) -> None:
        """Apply a single-qubit operator.

        Parameters
        ----------
        op : Matrix
            Matrix of shape :math:`(2, 2)` representing
            the operator to apply.
        qubit : int
            Target qubit index.
            qubit index
        """
        psi = tensordot(op, self.psi, (1, qubit))
        self.psi = np.moveaxis(psi, 0, qubit)

    @override
    def evolve(self, op: Matrix, qubits: Sequence[int]) -> None:
        """Apply a multi-qubit operator.

        Parameters
        ----------
        op : Matrix
            Matrix of shape :math:`(2^n, 2^n)` representing
            the operator to apply.
        qubits : Sequence[int]
            Target qubit indices.

        Notes
        -----
        This method is a fallback for circuit simulation and it's not required
        for pattern simulation.
        """
        op_dim = int(np.log2(len(op)))
        # TODO shape = (2,)* 2 * op_dim
        shape = [2 for _ in range(2 * op_dim)]
        op_tensor = op.reshape(shape)
        psi = tensordot(
            op_tensor,
            self.psi,
            (tuple(op_dim + i for i in range(len(qubits))), qubits),
        )
        self.psi = np.moveaxis(psi, range(len(qubits)), qubits)

    def dims(self) -> tuple[int, ...]:
        """Return the dimensions."""
        return self.psi.shape

    # Note that `@property` must appear before `@override` for pyright
    @property
    @override
    def nqubit(self) -> int:
        """Return the number of qubits."""
        return self.psi.ndim

    @override
    def remove_qubit(self, qubit: int) -> None:
        r"""Remove a separable qubit from the system and assemble a statevector for remaining qubits.

        This results in the same result as partial trace, if the qubit *qubit* is separable from the rest.

        For a statevector :math:`\ket{\psi} = \sum c_i \ket{i}` with sum taken over
        :math:`i \in [ 0 \dots 00,\ 0\dots 01,\ \dots,\
        1 \dots 11 ]`, this method returns

        .. math::
            \begin{align}
                \ket{\psi}' =&
                    c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 00}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 00} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 01}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 01} \\
                    & + c_{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k}}0_{\mathrm{k+1}} \dots 10}
                    \ket{0 \dots 0_{\mathrm{k-1}}0_{\mathrm{k+1}} \dots 10} \\
                    & + \dots \\
                    & + c_{1 \dots 1_{\mathrm{k-1}}0_{\mathrm{k}}1_{\mathrm{k+1}} \dots 11}
                    \ket{1 \dots 1_{\mathrm{k-1}}1_{\mathrm{k+1}} \dots 11},
           \end{align}

        (after normalization) for :math:`k =` qubit. If the :math:`k` th qubit is in :math:`\ket{1}` state,
        above will return zero amplitudes; in such a case the returned state will be the one above with
        :math:`0_{\mathrm{k}}` replaced with :math:`1_{\mathrm{k}}` .

        .. warning::
            This method assumes the qubit with index *qubit* to be separable from the rest,
            and is implemented as a significantly faster alternative for partial trace to
            be used after single-qubit measurements.
            Care needs to be taken when using this method.
            Checks for separability will be implemented soon as an option.

        Parameters
        ----------
        qubit : int
            qubit index
        """
        norm = _norm(self.psi)
        if isinstance(norm, SupportsFloat):
            assert not np.isclose(norm, 0)
        index: list[slice[int] | int] = [slice(None)] * self.psi.ndim
        index[qubit] = 0
        psi = self.psi[tuple(index)]
        norm = _norm(psi)
        if isinstance(norm, SupportsFloat) and math.isclose(norm, 0):
            index[qubit] = 1
            psi = self.psi[tuple(index)]
        self.psi = psi
        self.normalize()

    @override
    def entangle(self, qubits: tuple[int, int]) -> None:
        """Apply a CZ gate on two qubits.

        Parameters
        ----------
        qubits : tuple[int, int]
            (control, target) qubit indices.
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = tensordot(CZ_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    def tensor(self, other: Statevector) -> None:
        r"""Tensor product state with other qubits.

        Results in self :math:`\otimes` other.

        Parameters
        ----------
        other : :class:`graphix.sim.statevec.Statevector`
            statevector to be tensored with self
        """
        psi_self = self.psi.flatten()
        psi_other = other.psi.flatten()
        if psi_self.dtype == np.object_ and psi_other.dtype != np.object_:
            psi_other = psi_other.astype(np.object_, copy=False)  # pragma: nocover

        total_num = len(self.dims()) + len(other.dims())
        self.psi = kron(psi_self, psi_other).reshape((2,) * total_num)

    def cnot(self, qubits: tuple[int, int]) -> None:
        """Apply CNOT.

        Parameters
        ----------
        qubits : tuple of int
            (control, target) qubit indices
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = tensordot(CNOT_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Apply SWAP gate between two qubits.

        Parameters
        ----------
        qubits : tuple[int, int]
            (control, target) qubit indices.
        """
        # contraction: 2nd index - control index, and 3rd index - target index.
        psi = tensordot(SWAP_TENSOR, self.psi, ((2, 3), qubits))
        # sort back axes
        self.psi = np.moveaxis(psi, (0, 1), qubits)

    def normalize(self) -> None:
        """Normalize the state in-place."""
        # Note that the following calls to `astype` are guaranteed to
        # return the original NumPy array itself, since `copy=False` and
        # the `dtype` matches. This is important because the array is
        # then modified in place.
        if self.psi.dtype == np.object_:
            psi_o = self.psi.astype(np.object_, copy=False)
            norm_o = _norm_symbolic(psi_o)
            psi_o /= norm_o
            self.psi = psi_o
        else:
            psi_c = self.psi.astype(np.complex128, copy=False)
            norm_c = _norm_numeric(psi_c)
            psi_c /= norm_c
            self.psi = psi_c

    def flatten(self) -> Matrix:
        """Return flattened statevector."""
        return self.psi.flatten()

    @override
    def expectation_single(self, op: Matrix, qubit: int) -> complex:
        """Return the expectation value of single-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2*2 operator
        qubit : int
            target qubit index

        Returns
        -------
        complex : expectation value.
        """
        st1 = copy.copy(self)
        st1.normalize()
        st2 = copy.copy(st1)
        st1.evolve_single(op, qubit)
        return complex(np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten()))

    def expectation_value(self, op: Matrix, qubits: Sequence[int]) -> complex:
        """Return the expectation value of multi-qubit operator.

        Parameters
        ----------
        op : numpy.ndarray
            2^n*2^n operator
        qubits : list of int
            target qubit indices

        Returns
        -------
        complex : expectation value
        """
        st2 = copy.copy(self)
        st2.normalize()
        st1 = copy.copy(st2)
        st1.evolve(op, qubits)
        return complex(np.dot(st2.psi.flatten().conjugate(), st1.psi.flatten()))

    @override
    def replace_parameter(
        self, variable: Parameter, substitute: ExpressionOrSupportsFloat, *, copy: bool = False
    ) -> Statevector:
        psi = np.vectorize(lambda value: with_parameter(value, variable, substitute))(self.psi)
        if copy:
            result = Statevector()
            result.psi = np.vectorize(lambda value: with_parameter(value, variable, substitute))(self.psi)
            return result
        self.psi = psi
        return self

    @override
    def replace_parameters(
        self, assignment: Mapping[Parameter, ExpressionOrSupportsFloat], *, copy: bool = False
    ) -> Statevector:
        psi = np.vectorize(lambda value: with_parameters(value, assignment))(self.psi)
        if copy:
            result = Statevector()
            result.psi = np.vectorize(lambda value: with_parameters(value, assignment))(self.psi)
            return result
        self.psi = psi
        return self

    @override
    def permute(self, permutation: Sequence[int]) -> None:
        _check_permutation(permutation, self.nqubit)
        self.psi = np.transpose(self.psi, permutation)


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevector]):
    """MBQC simulator with statevector method."""

    state: Statevector = dataclasses.field(init=False, default_factory=lambda: Statevector(nqubit=0))


def _norm_symbolic(psi: npt.NDArray[np.object_]) -> ExpressionOrFloat:
    """Return norm of the state."""
    flat = psi.flatten()
    return check_expression_or_float(np.sqrt(np.sum(flat.conj() * flat)))


def _norm_numeric(psi: npt.NDArray[np.complex128]) -> float:
    flat = psi.flatten()
    norm_sq = np.sum(flat.conj() * flat)
    assert math.isclose(norm_sq.imag, 0, abs_tol=1e-15)
    return math.sqrt(norm_sq.real)


def _norm(psi: Matrix) -> ExpressionOrFloat:
    """Return norm of the state."""
    # Narrow psi to concrete dtype
    if psi.dtype == np.object_:
        return _norm_symbolic(psi.astype(np.object_, copy=False))
    return _norm_numeric(psi.astype(np.complex128, copy=False))


def _format_encoding(nqubit: int, i: int, encoding: _ENCODING) -> str:
    """Format the i-th basis vector as a ket. See :meth:`Statevector.to_dict` for additional details."""
    display_width = nqubit
    output = f"{i:0{display_width}b}"
    if encoding == "LSB":
        return output[::-1]
    return output
