"""Parameter class with symbolic computation using sympy

SympyParameter can be used in computation such as simulations.

"""

from __future__ import annotations

import numbers
from typing import Mapping

import numpy as np
import sympy as sp
from graphix.parameter import ExpressionOrComplex, ExpressionOrFloat, ExpressionWithTrigonometry, Parameter


class SympyExpression(ExpressionWithTrigonometry):
    """Expression with parameters.

    Implements arithmetic operations. This is essentially a wrapper over
    sp.Expr, exposing methods like cos, conjugate, etc., that are
    expected by the simulator back-ends.
    """

    def __init__(self, expression: sp.Expr) -> None:
        self._expression = expression

    def __mul__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(self._expression * other)
        elif isinstance(other, SympyExpression):
            return SympyExpression(self._expression * other._expression)
        else:
            return NotImplemented

    def __rmul__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(other * self._expression)
        elif isinstance(other, SympyExpression):
            return SympyExpression(other._expression * self._expression)
        else:
            return NotImplemented

    def __add__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(self._expression + other)
        elif isinstance(other, SympyExpression):
            return SympyExpression(self._expression + other._expression)
        else:
            return NotImplemented

    def __radd__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(other + self._expression)
        elif isinstance(other, SympyExpression):
            return SympyExpression(other._expression + self._expression)
        else:
            return NotImplemented

    def __sub__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(self._expression - other)
        elif isinstance(other, SympyExpression):
            return SympyExpression(self._expression - other._expression)
        else:
            return NotImplemented

    def __rsub__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(other - self._expression)
        elif isinstance(other, SympyExpression):
            return SympyExpression(other._expression - self._expression)
        else:
            return NotImplemented

    def __neg__(self) -> ExpressionOrFloat:
        return SympyExpression(-self._expression)

    def __truediv__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(self._expression / other)
        elif isinstance(other, SympyExpression):
            return SympyExpression(self._expression / other._expression)
        else:
            return NotImplemented

    def __rtruediv__(self, other) -> ExpressionOrFloat:
        if isinstance(other, numbers.Number):
            return SympyExpression(other / self._expression)
        elif isinstance(other, SympyExpression):
            return SympyExpression(other._expression / self._expression)
        else:
            return NotImplemented

    def __mod__(self, other) -> float:
        """mod magic function returns nan so that evaluation of
        mod of measurement angles in :meth:`graphix.pattern.is_pauli_measurement`
        will not cause error. returns nan so that this will not be considered Pauli measurement.
        """
        return np.nan

    def sin(self) -> ExpressionOrFloat:
        return SympyExpression(sp.sin(self._expression))

    def cos(self) -> ExpressionOrFloat:
        return SympyExpression(sp.cos(self._expression))

    def exp(self) -> ExpressionOrFloat:
        return SympyExpression(sp.exp(self._expression))

    def conjugate(self) -> ExpressionOrFloat:
        return SympyExpression(sp.conjugate(self._expression))

    def sqrt(self) -> ExpressionOrFloat:
        return SympyExpression(sp.sqrt(self._expression))

    @property
    def expression(self) -> sp.Expr:
        return self._expression

    def __repr__(self) -> str:
        return str(self._expression)

    def __str__(self) -> str:
        return str(self._expression)

    @staticmethod
    def __check_sympy_parameter(variable: Parameter) -> None:
        if not isinstance(variable, SympyParameter):
            raise ValueError(
                f"Sympy expressions can only be substituted with sympy parameters, not {variable.__class__}."
            )

    def subs(self, variable: Parameter, value: ExpressionOrFloat) -> ExpressionOrComplex:
        self.__check_sympy_parameter(variable)
        result = sp.N(self._expression.subs(variable._expression, value))
        if isinstance(result, numbers.Number) or not result.free_symbols:
            return complex(result)
        else:
            return SympyExpression(result)

    def xreplace(self, assignment: Mapping[Parameter, ExpressionOrFloat]) -> ExpressionOrComplex:
        for variable in assignment:
            self.__check_sympy_parameter(variable)
        sympy_assignment = {variable._expression: value for variable, value in assignment.items()}
        result = sp.N(self._expression.xreplace(sympy_assignment))
        if isinstance(result, numbers.Number) or not result.free_symbols:
            return complex(result)
        else:
            return SympyExpression(result)


class SympyParameter(Parameter, SympyExpression):
    """Placeholder for measurement angles, which allows the pattern optimizations
    without specifying measurement angles for measurement commands.
    Either use for rotation gates of :class:`Circuit` class or for
    the measurement angle of the measurement commands to be added with :meth:`Pattern.add` method.
    Example:
    .. code-block:: python

        import numpy as np
        from graphix import Circuit

        circuit = Circuit(1)
        alpha = Parameter("alpha")
        # rotation gate
        circuit.rx(0, alpha)
        pattern = circuit.transpile()
        # Both simulations (numeric and symbolic) will use the same
        # seed for random number generation, to ensure that the
        # explored branch is the same for the two simulations.
        seed = np.random.integers(2**63)
        # simulate with parameter assignment
        sv = pattern.subs(alpha, 0.5).simulate_pattern(pr_calc=False, rng=np.random.default_rng(seed))
        # simulate without pattern assignment
        # (the resulting state vector is symbolic)
        # Note: pr_calc=False is mandatory since we cannot compute probabilities on
        # symbolic states; we explore one arbitrary branch.
        sv2 = pattern.simulate_pattern(pr_calc=False, rng=np.random.default_rng(seed))
        # Substituting alpha in the resulting state vector should yield the same result
        assert np.allclose(sv.psi, sv2.subs(alpha, 0.5).psi)
    """

    def __init__(self, name: str) -> None:
        """Create a new :class:`Parameter` object.

        Parameters
        ----------
        name : str
            name of the parameter, used for binding values.
        """
        self._name = name
        super().__init__(sp.Symbol(name=name))

    @property
    def name(self) -> str:
        return self._name
