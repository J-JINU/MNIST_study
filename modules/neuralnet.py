import sys, os
import pickle

import numpy as np
# cupy_enable = True
# try:
#     import cupy as np
# except ImportError:
#     import numpy as np
#     cupy_enable = False


# print('__file__ ::', __file__)
# print('os.path.dirname(__file__) ::', os.path.dirname(__file__))
# print('os.path.abspath(os.path.dirname(__file__)) ::', os.path.abspath(os.path.dirname(__file__)))
# print('os.path.dirname(os.path.abspath(os.path.dirname(__file__))) ::', os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# print('os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))) ::', os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
# print('os.path.dirname(os.path.dirname(__file__)) ::', os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

from deep_study.dataset.mnist import load_mnist
from function import sigmoid, softmax, cross_entropy_error


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def get_data_all(normalize=True, flatten=True, one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=normalize, flatten=flatten, one_hot_label=one_hot_label)
    return x_train, t_train, x_test, t_test

def init_network():
    # need set directory(run .py's dir) in PYTHON terminal
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # if cupy_enable:
    #     W1 = np.ndarray(W1)
    #     W2 = np.ndarray(W2)
    #     W3 = np.ndarray(W3)
    #     b1 = np.ndarray(b1)
    #     b2 = np.ndarray(b2)
    #     b3 = np.ndarray(b3)
        
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) # pkl파일에 저장된 것 자체가 numpy array의 array인듯함 추후 cupy로 돌려 보던지 아니면 해당부분은 단지 weight라서 무시할지 생각하겠음
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) +b3
    return softmax(a3)

class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)
        
    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss
    
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        