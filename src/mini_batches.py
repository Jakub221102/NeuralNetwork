import numpy as np
from sklearn.utils import shuffle


def generate_mini_batches(dataset: np.ndarray, batch_size: int):
    x, y = shuffle(dataset[0], dataset[1])
    assert len(x) == len(y)
    batch_n = len(x) // batch_size
    for i in range(batch_n):
        yield x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]

