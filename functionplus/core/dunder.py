from functools import wraps

from . import helper
from . import types as ftypes

__all__ = ["binary_calls", "binary_dunder", "unary_dunder"]

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


def binary_dunder(op: ftypes.BinaryOperator, reverse: bool = True):
    """Decorator factory that converts a binary operator into
    a dunder method to be added to the Function class.

    Args:
        op (ftypes.BinaryOperator): The binary operator from the
        operator module to convert to a dunder method.
        reverse (bool, optional): Whether to return the reversed
        version of the dunder method as well, e.g. __mul__ and
        __rmul__. Defaults to True.

    Returns:
        __op__ (Callable[[Function, FunctionType | Any], Function]):
        The dunder method to be added.
        __rop__ (Callable[[Function, FunctionType | Any], Function]):
        The reversed dunder method to be added if reverse is True.
    """

    symbol = helper.operator_symbols[op.__name__]

    # decorates the operator
    @wraps(op, assigned=("__name__", "__doc__"))
    def __op__(self, other: ftypes.GenericFunction | ftypes.Any):
        def wrapper(*args: ftypes.Any, **kwargs: ftypes.Any) -> ftypes.Any:
            return op(*binary_calls(self, other, *args, **kwargs))

        # updates the wrapper's name and docstring appropriately
        f_name = helper.get_funcname(self)
        g_name = helper.get_funcname(other)

        new_name = f"({f_name} {symbol} {g_name})"
        wrapper.__name__ = new_name
        wrapper.__doc__ = f"Computes {new_name}(...)."

        # wraps the wrapper in the Function class if possible
        try:
            return self.__class__(wrapper)
        except TypeError:
            try:
                return other.__class__(wrapper)
            except TypeError:
                return wrapper

    # updates the dunder method's doctring
    newdoc = helper.operator_doc(op)
    __op__.__doc__ = newdoc

    if not reverse:
        return __op__

    # if reverse, do the same thing for the reversed version
    @wraps(op, assigned=("__name__", "__doc__"))
    def __rop__(self, other: ftypes.GenericFunction | ftypes.Any):
        return __op__(self=other, other=self)

    rnewdoc = helper.operator_doc(op, "other", "self")
    __rop__.__doc__ = rnewdoc

    return __op__, __rop__


def unary_dunder(op: ftypes.UnaryOperator):
    """Decorator factory that converts a unary operator into
    a dunder method to be added to the Function class.

    Args:
        op (ftypes.UnaryOperator): The unary operator from the
        operator module to convert to a dunder method.

    Returns:
        __op__ (Callable[[Function], Function]): The dunder method to be added.
    """

    symbol = helper.operator_symbols[op.__name__]

    # decorates the operator
    @wraps(op, assigned=("__name__", "__doc__"))
    def __op__(self):
        def wrapper(*args, **kwargs):
            return op(self(*args, **kwargs))

        # updates the operator's name and docstring appropriately
        f_name = helper.get_funcname(self)

        new_name = f"({symbol} {f_name})"
        wrapper.__name__ = new_name
        wrapper.__doc__ = f"Computes {new_name}(...)."

        # wraps the wrapper in the Function class
        try:
            return self.__class__(wrapper)
        except TypeError:
            return wrapper

    # updates the dunder method's doctring
    newdoc = helper.operator_doc(op)
    __op__.__doc__ = newdoc

    return __op__
