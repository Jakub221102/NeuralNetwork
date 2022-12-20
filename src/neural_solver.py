import numpy as np

from src.gradient.gradient_solver import GradientSolver


class NeuralNetworkSolver:
    def __init__(self, neurons_per_layer, layers, activation_function, cost_function, gradient_step, gradient_precision, gradient_max_iterations):
        self.parameters = {
            "neurons_per_layer": neurons_per_layer,
            "layers": layers,
            "activation_function": activation_function,
            "cost_function": cost_function,
        }
        self.gradient_solver = GradientSolver(self, gradient_step, gradient_precision, gradient_max_iterations)

    def get_parameter(self, param):
        return self.parameters[param] if param in self.parameters else None

    def train(self, train_dataset: np.array):
        """
        Train the neural network
        :param train_dataset: Train dataset
        :return: None
        """
        pass

    def predict(self, test_dataset_x) -> np.array:
        """
        Predict classes for test dataset
        :param test_dataset_x: Test dataset
        :return: Array of numbers (0-9) representing the predicted classes
        """
        pass

