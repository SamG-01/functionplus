from __future__ import annotations

import operator
from functools import partial, update_wrapper
from inspect import signature

from numpy import full_like

from .core import dunder, ops
from .core import types as ftypes

__all__ = ["Function"]


class Function:
    """A wrapper class for functions that facilitates
    function arithmetic, boolean logic, and composition.

    Attributes:
        function (ftypes.GenericFunction): The function being wrapped.
        components (set[ftypes.GenericFunction]): A set containing
        all of the functions used to create the total function.
        name: The function's name.

    """

    def __init__(
        self, function: ftypes.GenericFunction | ftypes.Self, name: str | None = None
    ) -> None:
        """Wraps the supplied function.

        Args:
            function (ftypes.GenericFunction | Function): Either a function
            or a pre-existing Function object.
            name (str | None, optional): What to call the wrapped
            function. Defaults to None.

        Raises:
            TypeError: If the function argument isn't callable.
        """

        if isinstance(function, Function):
            if name is None:
                self.name = function.name
            function = function.function

        if not callable(function):
            raise TypeError(f"{function} is not callable")

        update_wrapper(self, function)

        self.function: ftypes.GenericFunction = function
        if name is not None:
            self.name = name

        self.components = set([self.function])

    @property
    def name(self) -> str:
        """The function's name."""

        return self.__name__

    @name.setter
    def name(self, new_name: str) -> None:
        self.__name__ = new_name

    def __repr__(self) -> str:
        return f"<Function '{self.name}'>"

    def __hash__(self) -> int:
        return hash(self.function)

    def __call__(self, *args, **kwargs):
        """Calls self.function(*args, **kwargs)."""

        return self.function(*args, **kwargs)

    def __matmul__(self, other: ftypes.GenericFunction | ftypes.Any) -> ftypes.Self:
        """Composes the function with another Function."""

        f = getattr(self, "function", self)
        f_name = ops.get_funcname(self)

        g = getattr(other, "function", other)
        g_name = ops.get_funcname(other)

        # if g is a constant, treat
        # f @ g as a function call f(g)
        if not callable(g):
            return f(g)

        # otherwise, create a wrapper that
        # composes f and g
        def wrapper(*args, **kwargs):
            return f(g(*args, **kwargs))

        new_name = f"{g_name}; {f_name}"
        wrapper.__name__ = new_name
        wrapper.__doc__ = f"Applies the functions {new_name} from left to right."

        try:
            return self.__class__(wrapper)
        except TypeError:
            return other.__class__(wrapper)

    def __rmatmul__(self, other: ftypes.Self | ftypes.Any) -> ftypes.Self:
        """Composes another Function with this function."""

        if not callable(other):
            # if other is a constant, treat it as a function
            # that returns that function in the same shape as
            # whatever input it receives
            def other_func(x):
                dtype = getattr(other, "dtype", getattr(x, "dtype", None))
                try:
                    out = full_like(x, other, dtype=dtype)
                except ValueError:
                    out = full_like(x, other, dtype="object")
                if not out.ndim:
                    return out.item()
                return out

            other_func = Function(other_func, ops.get_funcname(other))
        else:
            other_func = other

        return self.__class__.__matmul__(other_func, self)

    def composed(self, n: int = 1) -> ftypes.Self:
        """Returns the composition of f with itself a total of n times."""

        if n < 0:
            raise ValueError("n must be a non-negative integer")

        if n == 0:
            return self.id()

        g = self
        for _ in range(n - 1):
            g @= self

        return g

    def partial(self, *pargs: ftypes.Any, **pkwargs: ftypes.Any) -> ftypes.Self:
        """Returns a new instance with pargs and pkwargs
        always applied to this function via functools.partial."""

        f_new = partial(self.function, *pargs, **pkwargs)
        return self.__class__(f_new)

    # adds arithmetic and boolean operators to the class
    for op_name in dir(operator):
        _op = getattr(operator, op_name)

        # if the operator isn't one of the ones we want,
        # skip it in the loop
        if not callable(_op) or _op.__name__ not in ops.operator_symbols:
            continue

        # checks if the operator is unary
        if len(signature(_op).parameters) == 1:
            try:
                locals()[op_name] = dunder.unary_dunder(_op)
            except NotImplementedError:
                #print(f"Unary operator {op_name} couldn't be added to Function")
                pass
            continue

        # otherwise, treat it as a binary one
        try:
            op__, rop__ = dunder.binary_dunder(_op)
        except NotImplementedError:
            print(f"Binary operator {op_name} couldn't be added to Function")
            continue

        locals()[op_name] = op__

        # adds __rop__ as well if it exists in general
        rop_name = op_name[:2] + "r" + op_name[2:]
        if hasattr(float, rop_name):
            locals()[rop_name] = rop__

    # removes temporary variables
    del op_name, _op, op__, rop__, rop_name  # pylint: disable=W0631

    @classmethod
    def id(cls, name: str = "id") -> ftypes.Self:
        """Returns the identity function."""

        return cls(lambda x: x, name)
