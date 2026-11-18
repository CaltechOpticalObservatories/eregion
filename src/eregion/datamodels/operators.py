"""
Arithmetic operators shared by Output, DetImage and FocalPlaneImage.
"""
from numbers import Number

import numpy as np


class ArithmeticOperators:
    """
    Mixin giving `+`, `-`, `*` and `/` (and their reflected forms, e.g. `1 - x`, `2 / x`) to a class that implements
    `_operate(op, other, reflected)`, where op is one of 'add', 'subtract', 'multiply', 'divide'. `_operate` returns
    NotImplemented for operands it does not support, so Python can try the other operand or raise TypeError.
    """

    # Make numpy defer to the reflected operators instead of broadcasting over this object as if it were a sequence,
    # so that e.g. `np.float64(2) * image` calls image.__rmul__.
    __array_ufunc__ = None

    def _operate(self, op: str, other, reflected: bool = False):
        raise NotImplementedError

    @staticmethod
    def _is_operand(other) -> bool:
        """ True for the non-image operands the operators accept: numbers and numpy arrays. """
        return isinstance(other, (Number, np.ndarray))

    def __add__(self, other):
        return self._operate('add', other)

    def __radd__(self, other):
        return self._operate('add', other, reflected=True)

    def __sub__(self, other):
        return self._operate('subtract', other)

    def __rsub__(self, other):
        return self._operate('subtract', other, reflected=True)

    def __mul__(self, other):
        return self._operate('multiply', other)

    def __rmul__(self, other):
        return self._operate('multiply', other, reflected=True)

    def __truediv__(self, other):
        return self._operate('divide', other)

    def __rtruediv__(self, other):
        return self._operate('divide', other, reflected=True)
