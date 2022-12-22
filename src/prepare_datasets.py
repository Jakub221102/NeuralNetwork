import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

MNIST_PATH = "data/mnist/"
TEST_SIZE = 0.2
IMAGE_PATH = "images/"


def load_mnist(path, kind='train'):
    """Load MNIST data from path"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def prepare_datasets():
    """Prepare train and test datasets"""
    X, y = load_mnist(MNIST_PATH, kind='train')
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, y, test_size=TEST_SIZE, random_state=0)
    X_test, Y_test = load_mnist(MNIST_PATH, kind='t10k')
    X_train = X_train/255
    X_validate = X_validate/255
    X_test = X_test/255
    Y_train = [convert_label_to_neuron_values(y) for y in Y_train]
    X_train = [np.reshape(x, (784, 1)) for x in X_train]
    train_data = (X_train, Y_train)
    X_validate = [np.reshape(x, (784, 1)) for x in X_validate]
    validate_data = (X_validate, Y_validate)
    X_test = [np.reshape(x, (784, 1)) for x in X_test]
    test_data = (X_test, Y_test)

    return train_data, validate_data, test_data


def convert_label_to_neuron_values(label: int) -> np.ndarray:
    neuron_values = np.zeros((10, 1))
    neuron_values[label] = 1.0
    return neuron_values


if __name__ == '__main__':
    train_dataset, validate_dataset, test_dataset = prepare_datasets()
    # train_dataset = list(train_dataset)
    # print(train_dataset[0])
    # print(train_dataset[0][0])
    # print(train_dataset[1][0])
    # print(validate_dataset[0].shape)
    # generate_img(train_dataset[0], train_dataset[1])
