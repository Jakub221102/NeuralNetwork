from typing import List

import numpy as np
from sklearn import metrics
from math import sqrt
from src.tool_function.activation_function import ActivationFunction, SigmoidActivationFunction
from src.tool_function.cost_function import CostFunction, QuadraticCostFunction
from src.mini_batches import generate_mini_batches
from src.utils import add_matrices_in_place, convert_label_to_neuron_values


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
        self.weights = [np.random.uniform(-1 / sqrt(x), 1 / sqrt(x), [y, x]) for x, y in
                        zip(neurons_per_layer[:-1], neurons_per_layer[1:])]
        self.neuron_values = [np.zeros(y, 1) for y in neurons_per_layer]
        self.activations = [np.zeros(y, 1) for y in neurons_per_layer]

    def get_parameter(self, param):
        return self.parameters[param] if param in self.parameters else None

    def train(self, train_dataset: np.array):
        """
        Train the neural network
        :param train_dataset: Train dataset
        :return: None
        """
        for i in range(self.get_parameter("epochs")):
            print("Epoch: ", i + 1)
            for X, Y in generate_mini_batches(train_dataset, self.get_parameter("batch_size")):
                self._stochistic_gradient(X, Y)

    def _stochistic_gradient(self, X, Y):
        """
        One iteration of stochistic gradient descent
        :param x: Input
        :param y: Output
        :return: None
        """
        weights_gradient = [np.full_like(w, 0) for w in self.weights]
        biases_gradient = [np.full_like(b, 0) for b in self.biases]
        for x, y in zip(X, Y):
            self._update_network_state(x)
            weights_diff, biases_diff = self._backpropagation(convert_label_to_neuron_values(y))
            add_matrices_in_place(weights_gradient, weights_diff)
            add_matrices_in_place(biases_gradient, biases_diff)
        add_matrices_in_place(self.weights,
                              [self.get_parameter("gradient_step") * w / len(X) for w in weights_gradient])
        add_matrices_in_place(self.biases, [self.get_parameter("gradient_step") * b / len(X) for b in biases_gradient])

    def _update_network_state(self, x):
        """
        Update the state of the network
        :param x: Input
        :return: None
        """
        self.activations[0] = x
        for i, b, w in zip(range(len(self.biases)), self.biases, self.weights):
            self.neuron_values[i + 1] = np.dot(w, self.activations[i]) + b
            self.activations[i + 1] = self.get_parameter("activation_function").get_value(self.neuron_values[i + 1])

    def _backpropagation(self, y):
        """
        Backpropagation algorithm
        :param x: Input
        :param y: Output
        :return: None
        """
        weights_diff = [np.full_like(w, 0) for w in self.weights]
        biases_diff = [np.full_like(b, 0) for b in self.biases]
        cost_derivative = self.get_parameter("cost_function").get_derivative(self.activations[-1], y)
        activation_derivative = self.get_parameter("activation_function").get_derivative(self.neuron_values[-1])
        delta = cost_derivative * activation_derivative
        biases_diff[-1] = delta
        weights_diff[-1] = np.dot(delta.reshape((len(delta), 1)), self.activations[-2].reshape((1, len(self.activations[-2]))))
        for i in range(2, self.get_parameter("layers")):
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * self.get_parameter(
                "activation_function").get_derivative(self.neuron_values[-i])
            biases_diff[-i] = delta
            weights_diff[-i] = np.dot(delta.reshape((len(delta), 1)), self.activations[-i - 1].reshape((1, len(self.activations[-i - 1]))))
        return weights_diff, biases_diff

    def predict(self, test_dataset_x) -> np.array:
        """
        Predict classes for test dataset
        :param test_dataset_x: Test dataset
        :return: Array of numbers (0-9) representing the predicted classes
        """
        predictions = []
        for i in range(len(test_dataset_x)):
            input = test_dataset_x[i]
            for ii, b, w in zip(range(len(self.biases)), self.biases, self.weights):
                input = self.parameters["activation_function"].get_value(np.dot(w, input) + b)
            predictions.append(np.argmax(input))
        return predictions

    @staticmethod
    def accuracy(predictions, test_dataset):
        # good_number = 0
        #     for predicted, y_value in zip(predictions, test_dataset[1]):
        #         if predicted == y_value:
        #             good_number += 1
        #     precent = round(good_number * 100 / test_dataset[1].size, 2)
        return metrics.accuracy_score(test_dataset[1], predictions)

