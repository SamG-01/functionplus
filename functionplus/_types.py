from typing import Any, Callable, Self

GenericFunction = Callable[..., Any]
BinaryOperator = Callable[[Any, Any], Any]
UnaryOperator = Callable[[Any], Any]
Operator = BinaryOperator | UnaryOperator

__all__ = [
    "Any",
    "Callable",
    "Self",
    "GenericFunction",
    "BinaryOperator",
    "UnaryOperator",
    "Operator",
]
