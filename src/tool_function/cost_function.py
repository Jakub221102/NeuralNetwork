from abc import abstractmethod

import numpy as np


class CostFunction:
    @abstractmethod
    def get_value(self, x: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def get_derivative(self, x: np.ndarray, y: np.ndarray):
        ...


class QuadraticCostFunction(CostFunction):
    def get_value(self, x: np.ndarray, y: np.ndarray):
        return np.sum((x - y)**2)

    def get_derivative(self, x: np.ndarray, y: np.ndarray):
        return 2 * (x - y)

