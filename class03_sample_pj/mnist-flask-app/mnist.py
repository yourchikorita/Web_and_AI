import os
import gzip
import _pickle as cPickle
import wget
import numpy as np

# from https://github.com/kdexd/digit-classifier
seed = 11768

def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz', out='data')

    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
    training_data, validation_data, test_data = cPickle.load(data_file, encoding='latin1')
    data_file.close()

    np.random.seed(seed)

    indices = np.arange(training_data[0].shape[0])
    np.random.shuffle(indices)

    train_x = training_data[0][indices]
    train_y = vectorized_result(training_data[1][indices])


    val_x = validation_data[0]
    val_y = validation_data[1]

    test_x = test_data[0]
    test_y = test_data[1]
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def vectorized_result(y):
    b = np.zeros((y.size, y.max()+1))
    b[np.arange(y.size),y] = 1
    return b
