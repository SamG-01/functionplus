from typing import Any, Callable, Self

FunctionType = Callable[..., Any]
BinaryOperator = Callable[[Any, Any], Any]
UnaryOperator = Callable[[Any], Any]
Operator = BinaryOperator | UnaryOperator

__all__ = [
    "Any",
    "Callable",
    "Self",
    "FunctionType",
    "BinaryOperator",
    "UnaryOperator",
    "Operator",
]
