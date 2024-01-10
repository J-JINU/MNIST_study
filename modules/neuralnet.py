import sys, os
import pickle

import numpy as np
cupy_enable = True
try:
    import cupy as np
except ImportError:
    import numpy as np
    cupy_enable = False

# print('__file__ ::', __file__)
# print('os.path.dirname(__file__) ::', os.path.dirname(__file__))
# print('os.path.abspath(os.path.dirname(__file__)) ::', os.path.abspath(os.path.dirname(__file__)))
# print('os.path.dirname(os.path.abspath(os.path.dirname(__file__))) ::', os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# print('os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) ::', os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
# print('os.path.dirname(os.path.dirname(__file__)) ::', os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

from deep_study.dataset.mnist import load_mnist
from function import *


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def get_data_all(normalize=True, flatten=True, one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=normalize, flatten=flatten, one_hot_label=one_hot_label)
    return x_train, t_train, x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) +_b3
    return softmax(a3)