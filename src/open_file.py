import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split



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

X, y = load_mnist('mnist/', kind='train')


X_train, X_test_validate, y_train, y_test_validate = train_test_split(
            X, y, test_size=0.2
        )
print(X_train.shape)

print(X_test_validate.shape)

X_test, y_test = load_mnist('mnist/', kind='t10k')
print(X_test.shape)


fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X[y == 4][i].reshape(28, 28)
    ax[i].imshow(img, cmap = 'Greys', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
fig.tight_layout()
fig.savefig("images/cos")



