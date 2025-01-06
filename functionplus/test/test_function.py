import operator

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..function import Function


class TestFunction:
    @pytest.fixture(
        scope="class", params=[*np.linspace(1, 10), np.arange(11, 20).reshape(3, 3)]
    )
    def inputs(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture(scope="class")
    def x(self) -> Function:
        return Function.id("x")

    @pytest.fixture(scope="class", params=[(1, 1), (2, 3), (-1, 2)])
    def f(self, request: pytest.FixtureRequest, x: Function) -> Function:
        a, b = request.param
        return a * x**b

    @pytest.fixture(scope="class", params=[np.log, np.arctan, np.cos, np.exp])
    def g(self, request: pytest.FixtureRequest, x: Function) -> Function:
        return Function(request.param)

    def test_composition(self, f: Function, g: Function, inputs) -> None:
        assert_allclose((f @ g)(inputs), f(g(inputs)))

    @pytest.mark.parametrize(
        "op_name", ["add", "floordiv", "mul", "mod", "pow", "sub", "truediv"]
    )
    def test_arithmetic(self, f: Function, g: Function, inputs, op_name: str) -> None:
        op = getattr(operator, op_name)
        assert_allclose(op(f, g)(inputs), op(f(inputs), g(inputs)))

    @pytest.mark.parametrize("op_name", ["eq", "ge", "gt", "le", "lt", "ne"])
    def test_compare(self, f: Function, g: Function, inputs, op_name: str) -> None:
        op = getattr(operator, op_name)
        assert np.all(op(f, g)(inputs) == op(f(inputs), g(inputs)))

    @pytest.mark.parametrize("op_name", ["and_", "eq", "ne", "or_", "xor"])
    def test_boolean(self, f: Function, g: Function, inputs, op_name: str) -> None:
        op = getattr(operator, op_name)
        assert np.all(
            (op(f >= 5, g <= 5))(inputs) == (op(f(inputs) >= 5, g(inputs) <= 5))
        )

    @pytest.mark.parametrize("op_name", ["abs", "neg", "pos"])
    def test_unary(self, f: Function, inputs, op_name: str) -> None:
        op = getattr(operator, op_name)
        assert_allclose(op(f)(inputs), op(f(inputs)))

    def test_components(self, f: Function, g: Function, x: Function) -> None:
        h = 2 + f @ f + g @ g + np.cos
        assert (
            h.components
            == set([*f.components, *g.components, np.cos])
            == set([x, g, np.cos])
            == set([x.function, g.function.np.cos])
        )

    def test_partial(self, inputs) -> None:
        h = Function(np.arctan2)
        h_p = h.partial(np.pi / 6)
        assert_allclose(h_p(inputs), h(np.pi / 6, inputs))
