from __future__ import annotations

from . import types as ftypes

__all__ = ["operator_symbols", "get_funcname", "operator_doc"]

operator_symbols = {
    "abs": "abs ",
    "add": "+",
    "eq": "==",
    "floordiv": "//",
    "ge": ">=",
    "gt": ">",
    "le": "<=",
    "lt": "<",
    "mod": "%",
    "mul": "*",
    "ne": "!=",
    "not_": "not ",
    "neg": "-",
    "or_": "or",
    "pos": "+",
    "pow": "**",
    "sub": "-",
    "truediv": "/",
    "xor": "^",
}


def get_funcname(obj: ftypes.GenericFunction | ftypes.Any) -> str:
    """Gets the __name__ of obj or the function it wraps."""

    name = getattr(obj, "__name__", None)
    if name is None:
        function = getattr(obj, "function", obj)
        name = getattr(function, "__name__", str(function))
    return name


def operator_doc(op: ftypes.Operator, a: str = "self", b: str = "other") -> str:
    """Replaces 'a' and 'b' in an operator's docstring with
    'self' and 'other' as appropriate.

    Args:
        op (Operator): The operator from the operator module.
        a (str, optional): The string to replace 'a' with in the docstring.
        Defaults to 'self'.
        b (str, optional): The string to replace 'b' with in the docstring.
        Defaults to 'other'.

    Raises:
        NotImplementedError: If the original string isn't in the
        expected format for 'Same as ...'.

    Returns:
        str: The docstring with the appropriate replacements made.
    """

    doc: str = op.__doc__
    if "Return" in doc:
        raise NotImplementedError(f"symbol is not available for {op}")

    for punc in (" ", ",", ".", ")"):
        doc = doc.replace("a" + punc, a + punc)
        doc = doc.replace("b" + punc, b + punc)

    return doc
