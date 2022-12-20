from abc import abstractmethod

import numpy as np


class ActivationFunction:
    @abstractmethod
    def get_value(self, x):
        ...

    @abstractmethod
    def get_derivative(self, x):
        ...


class ReLUActivationFunction(ActivationFunction):
    def get_value(self, x):
        return max(0, x)

    def get_derivative(self, x):
        return 1 if x > 0 else 0


class SigmoidActivationFunction(ActivationFunction):
    def get_value(self, x):
        return 1 / (1 + np.exp(-x))

    def get_derivative(self, x):
        return self.get_value(x) * (1 - self.get_value(x))

