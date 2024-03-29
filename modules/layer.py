# basic library
import sys, os
import csv
from collections import OrderedDict
# install library
import cupy as np
# self library



class TwoLayerNet:
    """_summary_
    TODO 추후 해당 클래스를 multi layer로 수정해야한다.
    TODO 하나의 클래스가 1개의 신경망을 만들 수 있도록 제작할 예정이다.
    TODO 각 layer의 input/output size에 문제가 없는지 검토하는 기능 필요
    TODO 일정 횟수마다 학습상태 log저장
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        
        self.layers = OrderedDict()
        # self.layers['Affine1'] = 

    def save_datalog(index, loss):
        with open("/home/jinu/Documents/project/MNIST_study/log/datalog.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([index, loss])