from typing import List

import numpy as np
from sklearn import metrics
from math import sqrt
from src.tool_function.activation_function import ActivationFunction
from src.tool_function.cost_function import CostFunction


class NeuralNetworkSolver:
    def __init__(self, neurons_per_layer: List[int], activation_function: ActivationFunction,
                 cost_function: CostFunction, gradient_step, batch_size, epochs):
        self.parameters = {
            "neurons_per_layer": neurons_per_layer,
            "layers": len(neurons_per_layer),
            "activation_function": activation_function,
            "cost_function": cost_function,
            "gradient_step": gradient_step,
            "batch_size": batch_size,
            "epochs": epochs
        }
        self.biases = [np.random.randn(y, 1) for y in neurons_per_layer[1:]]
        self.weights = [np.random.uniform(-1/sqrt(x), 1/sqrt(x), [y, x]) for x, y in zip(neurons_per_layer[:-1], neurons_per_layer[1:])]

    def get_parameter(self, param):
        return self.parameters[param] if param in self.parameters else None

    def _backpropagation(self, x, y):
        """
        Backpropagation algorithm
        :param x: Input
        :param y: Output
        :return: None
        """
        pass

    def _stochistic_gradient(self, x, y):
        """
        Stochistic gradient descent algorithm
        :param x: Input
        :param y: Output
        :return: None
        """
        pass

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
        predictions = []
        for i in range(len(test_dataset_x)):
            for b, w in zip(self.biases, self.weights):
                result_of_activation = self.parameters["activation_function"].get_value(np.dot(w, test_dataset_x[i])+b)
            predictions.append(np.argmax(result_of_activation))

        return predictions

    @staticmethod
    def accuracy(predictions, test_dataset):
        # good_number = 0
        #     for predicted, y_value in zip(predictions, test_dataset[1]):
        #         if predicted == y_value:
        #             good_number += 1
        #     precent = round(good_number * 100 / test_dataset[1].size, 2)
        return metrics.accuracy_score(test_dataset[1], predictions)


