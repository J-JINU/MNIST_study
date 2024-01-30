import os
import time
import polars as po
import matplotlib.pyplot as plt
from modules import neuralnet as nnet
from modules import function as func
from PIL import Image

import cupy as np
# cupy_enable = True
# try:
#     import cupy as np
# except ImportError:
#     import numpy as np
#     cupy_enable = False

from deep_study.dataset.mnist import load_mnist

# arr = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
# print(arr.shape)

arr = np.arange(0, 16)
arr = arr.reshape(4, 1, 1, -1) # -1입력하면 자동으로 4로 계산되어 들어감.
arr2 = arr.reshape(2, 1, 2, -1)
print(arr2)
print(arr2.shape)
arr2_origin_shape = arr2.shape
arr2 = arr2.reshape(arr2.shape[0], -1)
print(arr2)
print(arr2.shape)
arr2 = arr2.reshape(*arr2_origin_shape)
print(arr2)
print(arr2.shape)

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# x_train = np.asarray(x_train)
# t_train = np.asarray(t_train)
# x_test = np.asarray(x_test)
# t_test = np.asarray(t_test)

# train_loss_list = []

# iters_num = 10000
# train_size = x_train.shape[0]
# batch_size = 100
# learning_rate = 0.1
# network = nnet.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# pred_start = time.perf_counter()
# for i in range(iters_num):
#     print(f'start {i} iter')
#     start_time = time.perf_counter()
#     batch_mask = np.random.choice(train_size, batch_size)
#     x_batch = x_train[batch_mask]
#     t_batch = t_train[batch_mask]
    
#     grad = network.numerical_gradient(x_batch, t_batch)
    
#     for key in ('W1', 'b1', 'W2', 'b2'):
#         network.params[key] -= learning_rate * grad[key]
        
#     loss = network.loss(x_batch, t_batch)
#     train_loss_list.append(loss)
#     end_time = time.perf_counter()
#     print(f'end {i} iter, run time : {end_time - start_time}')
    
# pred_end = time.perf_counter()
# print(f'end pred, total time : {pred_end - pred_start}')
# # 그래프 그리기
# markers = {'train': 'o', 'test': 's'}
# x = np.arange(len(train_loss_list))
# plt.plot(x, train_loss_list, label='train loss')
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.ylim(0, 1.0)
# plt.legend(loc='lower right')
# plt.show()
