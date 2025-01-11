from abc import abstractmethod
from functools import wraps
import operator

from numpy import full_like

from . import ops
from . import types as ftypes


class DunderOperator:
    def __init__(
        self, name: str, op: ftypes.Operator | None = None, is_rop: bool = False
    ) -> None:
        if op is None:
            self._op: ftypes.Operator = getattr(operator, name)
        else:
            self._op = op
        if not callable(self._op):
            raise NotImplementedError("Provided operator is not a method")

        self.symbol = ops.operator_symbols[self._op.__name__]
        self.is_rop = is_rop

        if is_rop:
            if not name.startswith("__") and name.endswith("__"):
                raise ValueError(f"Method {self._op} is not a dunder method")
            self.name = name[:2] + "r" + name[2:]
            self.doc = ops.operator_doc(self._op, "other", "self")
        else:
            self.name = name
            self.doc = ops.operator_doc(self._op)

    @abstractmethod
    def op(self, cls: type) -> ftypes.GenericFunction: ...

    @staticmethod
    def get_components(instance: object):
        if not callable(instance):
            return set()
        return getattr(instance, "components", set([instance]))

    def __get__(self, instance: object, owner: type):
        return self.op(owner).__get__(instance, owner)


class DunderUnaryOperator(DunderOperator):
    def __init__(self, name: str, op: ftypes.Operator | None = None) -> None:
        super().__init__(name, op, False)
        del self.is_rop

    def op(self, cls: type):
        @wraps(self._op, assigned=("__name__", "__doc__"))
        def __op__(fself):
            def wrapper(*fargs, **fkwargs):
                return self._op(fself(*fargs, **fkwargs))

            # updates the operator's name and docstring appropriately
            f_name = ops.get_funcname(fself)

            new_name = f"({self.symbol} {f_name})"
            wrapper.__name__ = new_name
            wrapper.__doc__ = f"Computes {new_name}(...)."

            h = cls(wrapper)
            h.components = self.get_components(fself)
            return h

        __op__.__doc__ = self.doc
        return __op__


class DunderBinaryOperator(DunderOperator):
    @staticmethod
    def binary_calls(
        f: ftypes.GenericFunction | ftypes.Any,
        g: ftypes.GenericFunction | ftypes.Any,
        *args: ftypes.Any,
        **kwargs: ftypes.Any,
    ) -> tuple[ftypes.Any, ftypes.Any]:
        """Takes two input objects and calls with them with the
        arguments passed in alongside them, if possible.

        Args:
            f (GenericFunction | Any): The first object to be called if possible.
            g (GenericFunction | Any): The second object to be called if possible.
            args (Any): The positional arguments to pass into f and g.
            kwargs (Any): The keyword arguments to pass into f and g.

        Returns:
            tuple[Any, Any]: Any relevant function outputs.
        """

        cf, cg = callable(f), callable(g)
        if cf and cg:
            return f(*args, **kwargs), g(*args, **kwargs)
        if cf:
            return f(*args, **kwargs), g
        if cg:
            return f, g(*args, **kwargs)
        return f, g

    def op(self, cls: type):
        @wraps(self._op, assigned=("__name__", "__doc__"))
        def __op__(fself, fother):
            def wrapper(*fargs, **fkwargs):
                if self.is_rop:
                    calls = self.binary_calls(fother, fself, *fargs, **fkwargs)
                calls = self.binary_calls(fself, fother, *fargs, **fkwargs)
                return self._op(*calls)

            # updates the wrapper's name and docstring appropriately
            f_name = ops.get_funcname(fself)
            g_name = ops.get_funcname(fother)

            new_name = f"({f_name} {self.symbol} {g_name})"
            wrapper.__name__ = new_name
            wrapper.__doc__ = f"Computes {new_name}(...)."

            h = cls(wrapper)
            h.components = self.get_components(fself) | self.get_components(fother)
            return h

        __op__.__doc__ = self.doc
        return __op__


class CompositionOperator(DunderOperator):
    def __init__(self, is_rop: bool = False) -> None:  # pylint: disable=W0231
        self.is_rop = is_rop
        self.name = "__rmatmul__" if self.is_rop else "__matmul__"

    def op(self, cls: type):
        return self.rop(cls) if self.is_rop else self.lop(cls)

    def lop(self, cls: type):
        def __matmul__(fself, fother: ftypes.GenericFunction | ftypes.Any):
            """Composes the function with another Function."""

            f = getattr(fself, "function", fself)
            g = getattr(fother, "function", fother)

            # if g is a constant, treat
            # f @ g as a function call f(g)
            if not callable(g):
                return f(g)

            f_name = ops.get_funcname(fself)
            g_name = ops.get_funcname(fother)

            # otherwise, create a wrapper that
            # composes f and g
            def wrapper(*args, **kwargs):
                return f(g(*args, **kwargs))

            new_name = f"{g_name}; {f_name}"
            wrapper.__name__ = new_name
            wrapper.__doc__ = f"Applies the functions {new_name} from left to right."

            h = cls(wrapper)
            h.components = self.get_components(fself) | self.get_components(fother)
            return h

        return __matmul__

    def rop(self, cls: type):
        def __rmatmul__(self, other: ftypes.Self | ftypes.Any):
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

                other_func = cls(other_func, ops.get_funcname(other))
            else:
                other_func = other

            return self.lop(cls)(other, self)

        return __rmatmul__
