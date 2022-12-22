from typing import List

import numpy as np


def add_matrices_in_place(first_matrix: List[np.ndarray], second_matrix: List[np.ndarray]):
    for i in range(len(first_matrix)):
        first_matrix[i] += second_matrix[i]


def subtract_matrices_in_place(first_matrix: List[np.ndarray], second_matrix: List[np.ndarray]):
    for i in range(len(first_matrix)):
        first_matrix[i] -= second_matrix[i]


