import numpy as np

from src.tool_function.activation_function import ReLUActivationFunction, SigmoidActivationFunction
from src.tool_function.cost_function import basic_cost_function


def test_cost_1():
    x = np.array([1, 2, 3])
    y = np.array([1, 0, 0])
    assert basic_cost_function(x, y) == 13


def test_cost_2():
    x = np.array([1, 2, 3])
    y = np.array([0, 1, 0])
    assert basic_cost_function(x, y) == 11


def test_cost_3():
    x = np.array([1, 2, 3])
    y = np.array([0, 0, 1])
    assert basic_cost_function(x, y) == 9


def test_cost_float():
    x = np.array([0.5, 0, 0])
    y = np.array([1, 0, 0])
    assert basic_cost_function(x, y) == 0.25


def test_relu():
    func = ReLUActivationFunction()
    assert func.get_value(-1) == 0
    assert func.get_value(0) == 0
    assert func.get_value(1) == 1


def test_sigmoid():
    func = SigmoidActivationFunction()
    last_value = -1
    for x in np.arange(-256, 256.01, 0.01):
        value = func.get_value(x)
        assert value >= last_value
        assert -1 <= value <= 1
        last_value = value
