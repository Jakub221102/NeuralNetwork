import sys
import os
import matplotlib.pyplot as plt
import numpy as np 

sys.path.append(os.getcwd())
from src.prepare_datasets import prepare_datasets
from src.neural_solver import NeuralNetworkSolver
from src.tool_function.activation_function import ActivationFunction, SigmoidActivationFunction
from src.tool_function.cost_function import CostFunction, QuadraticCostFunction


def main():
    # Test config
    train_dataset, validate_dataset, test_dataset = prepare_datasets()

    lst = []
    lst2 = []
    lst3= []
    lst4= []
    for i in range(7):
        solver = NeuralNetworkSolver([784, 16, 16, 16, 16, 10], SigmoidActivationFunction(), QuadraticCostFunction(),  0.5, 10, i*5)
        solver.train(train_dataset)
        predictions = solver.predict(validate_dataset[0])
        accuracy = solver.accuracy(predictions, validate_dataset)
        lst.append(i*5)
        lst2.append(accuracy)

        solver2 = NeuralNetworkSolver([784, 16, 16, 16, 16, 10], SigmoidActivationFunction(), QuadraticCostFunction(),  0.5, 100, i*5)
        solver2.train(train_dataset)
        predictions2 = solver2.predict(validate_dataset[0])
        accuracy2 = solver2.accuracy(predictions2, validate_dataset)
        lst3.append(i*5)
        lst4.append(accuracy2)

    plt.plot(lst, lst2, linestyle="--", marker="o", label = "Batch 10")
    plt.plot(lst3, lst4, linestyle="--", marker="o", label = "Batch 100")
    plt.legend(loc="lower right")
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(min(lst), max(lst)+5, 5))
    plt.yticks(np.arange(0.00, 1.00, 0.05))
    plt.title('Accuracy for 4 layers with 16 neurons each \n Learning rate 0.5 , Batch size 10 and 100')
    plt.savefig("images/plot4.jpg")


if __name__ == '__main__':
    main()
