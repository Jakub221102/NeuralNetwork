import numpy as np


def basic_cost_function(x: np.array, y: np.array) -> float:
    return float(np.sum((x - y)**2))
