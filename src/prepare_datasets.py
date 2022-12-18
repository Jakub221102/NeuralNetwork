import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

MNIST_PATH = "../data/mnist/"
TEST_SIZE = 0.2
IMAGE_PATH = "../images/"


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

    return (X_train, Y_train), (X_validate, Y_validate), (X_test, Y_test)


def generate_img(X, Y):
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = X[Y == 4][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    fig.tight_layout()
    fig.savefig(IMAGE_PATH + "cos")


if __name__ == '__main__':
    train_dataset, validate_dataset, test_dataset = prepare_datasets()
    print(train_dataset[0].shape)
    print(validate_dataset[0].shape)
    print(test_dataset[0].shape)
    generate_img(train_dataset[0], train_dataset[1])
