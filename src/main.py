import sys
import os

sys.path.append(os.getcwd())
from src.prepare_datasets import prepare_datasets
from src.neural_solver import NeuralNetworkSolver
from src.tool_function.activation_function import ActivationFunction, SigmoidActivationFunction, ReLUActivationFunction
from src.tool_function.cost_function import CostFunction, QuadraticCostFunction


def main():
    # Test config
    train_dataset, validate_dataset, test_dataset = prepare_datasets()
    solver = NeuralNetworkSolver([784, 10], SigmoidActivationFunction(), QuadraticCostFunction(),  0.1, 1000, 20)
    params = solver.get_parameter("neurons_per_layer")
    print(params)
    print(solver.get_parameter("layers"))
    print(solver.get_parameter("activation_function"))
    print(solver.get_parameter("cost_function"))
    print(solver.get_parameter("gradient_step"))
    print(solver.get_parameter("batch_size"))
    print(solver.get_parameter("epochs"))
    solver.train(train_dataset)
    predictions = solver.predict(validate_dataset[0])
    accuracy = solver.accuracy(predictions, validate_dataset)
    print(accuracy)
    #print(train_dataset[0].shape)


if __name__ == '__main__':
    main()
