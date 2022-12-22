from typing import List

import numpy as np


def add_matrices_in_place(first_matrix: List[np.ndarray], second_matrix: List[np.ndarray]):
    for i in range(len(first_matrix)):
        first_matrix[i] += second_matrix[i]


def subtract_matrices_in_place(first_matrix: List[np.ndarray], second_matrix: List[np.ndarray]):
    for i in range(len(first_matrix)):
        first_matrix[i] -= second_matrix[i]


def convert_label_to_neuron_values(label: int) -> np.ndarray:
    neuron_values = np.zeros(10)
    neuron_values[label] = 1
    return neuron_values


if __name__ == "__main__":
    print(convert_label_to_neuron_values(2))
