import os
import time
import polars as po
import matplotlib.pyplot as plt
from modules import neuralnet as nnet
from modules import function as func
from PIL import Image

import numpy as np
# cupy_enable = True
# try:
#     import cupy as np
# except ImportError:
#     import numpy as np
#     cupy_enable = False

# def img_show(img):
#     pil_img = Image.fromarray(np.uint8(img))
#     pil_img.show()

# x_train, t_train, x_test, t_test = nnet.get_data_all(normalize=False)

# img = x_train[7]
# label = t_train[7]
# print(label)

# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)

# img_show(img)

# x, t = nnet.get_data()
# network = nnet.init_network()

# W1, W2, W3 = network['W1'], network['W2'], network['W3']
# print(x.shape)
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

# accuracy_cnt = 0
# start_time = time.perf_counter()
# for i in range(len(x)):
#     y = nnet.predict(network, x[i])
#     p = np.argmax(y)
#     if p == t[i]:
#         accuracy_cnt += 1
# end_time = time.perf_counter()        
# print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
# print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")


# batch_size = 100
# accuracy_cnt = 0
# start_time = time.perf_counter()
# for i in range(0, len(x), batch_size):
#     x_batch = x[i:i+batch_size]
#     y_batch = nnet.predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1) # predict result
#     accuracy_cnt += np.sum(p == t[i:i+batch_size]) #compare predict result to label
# end_time = time.perf_counter()   

# print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
# print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")

# x_train, t_train, x_test, t_test = nnet.get_data_all()

# y = np.array([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0], [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])
# t = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
# y = np.array([[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]])
# t = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
# y = np.array([[0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]])
# t = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
# print(func.cross_entropy_error(y, t))

# y = np.array([[0.1, 0.05, 0.6, 0.0, 0.05]])
# print(y)
# t = np.array([[2, 3, 0, 1, 4]])
# print(y[np.arange(y.shape[0]), t])

# class simpleNet:
#     def __init__(self):
#         self.W = np.random.randn(2,3) # 정규분포로 초기화

#     def predict(self, x):
#         return np.dot(x, self.W)

#     def loss(self, x, t):
#         z = self.predict(x)
#         y = func.softmax(z)
#         loss = func.cross_entropy_error(y, t)

#         return loss

x = np.array([0.6, 0.9,])
t = np.array([0, 0, 1])

net = nnet.simpleNet()

f = lambda w: net.loss(x, t)
dW = func.numerical_gradient(f, net.W)

print(dW)

# net = nnet.simpleNet()
# print(net.W)

# x = np.array([0.6, 0.9])
# p = net.predict(x)
# print(p)
# np.argmax(p)
# t = np.array([0, 0, 1])
# print(net.loss(x, t))
# f = lambda w: net.loss(x, t)
# dW = func.numerical_gradient(f, net.W)