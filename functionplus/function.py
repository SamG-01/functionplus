import operator
from functools import partial, update_wrapper
from inspect import signature

from .core import dunder, ops
from .core import types as ftypes

__all__ = ["Function"]

class Function:
    """A wrapper class for functions that facilitates
    function arithmetic, boolean logic, and composition.

    Attributes:
        function (ftypes.GenericFunction): The function being wrapped.
        components (set[ftypes.GenericFunction]): A set containing
        all of the functions used to create the total function. Does
        not include abs calls or operations involving non-callables.
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

    @classmethod
    def id(cls, name: str = "id") -> ftypes.Self:
        """Returns the identity function."""

        return cls(lambda x: x, name)

    @property
    def name(self) -> str:
        """The function's name."""

        return ops.get_funcname(self)

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

    __matmul__ = dunder.CompositionOperator(False)
    __rmatmul__ = dunder.CompositionOperator(True)


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
            __udunder__ = dunder.DunderUnaryOperator(op_name, _op)
            #locals()[__udunder__.name] = __udunder__
            setattr(Function, __udunder__.name, __udunder__)
        except (NotImplementedError, ValueError):
            #print(f"Unary operator {op_name} couldn't be added to Function")
            pass
        continue

    # otherwise, treat it as a binary one
    try:
        __bdunder__ = dunder.DunderBinaryOperator(op_name, _op)
        #locals()[__bdunder__.name] = __bdunder__
        setattr(Function, __bdunder__.name, __bdunder__)
    except (NotImplementedError, ValueError):
        #print(f"Binary operator {op_name} couldn't be added to Function")
        continue

    # adds __rop__ as well if it exists
    if not hasattr(float, op_name[:2] + "r" + op_name[2:]):
        continue
    try:
        __brdunder__ = dunder.DunderBinaryOperator(op_name, _op, True)
        #locals()[__brdunder__.name] = __brdunder__
        setattr(Function, __brdunder__.name, __brdunder__)
    except (NotImplementedError, ValueError):
        #print(f"Binary operator {rop_name} couldn't be added to Function")
        pass
