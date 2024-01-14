import os
import time
import polars as po
import matplotlib.pyplot as plt
from modules import neuralnet as net
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

# x_train, t_train, x_test, t_test = net.get_data_all(normalize=False)

# img = x_train[7]
# label = t_train[7]
# print(label)

# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)

# img_show(img)

# x, t = net.get_data()
# network = net.init_network()

# W1, W2, W3 = network['W1'], network['W2'], network['W3']
# print(x.shape)
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

# accuracy_cnt = 0
# start_time = time.perf_counter()
# for i in range(len(x)):
#     y = net.predict(network, x[i])
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
#     y_batch = net.predict(network, x_batch)
#     p = np.argmax(y_batch, axis=1) # predict result
#     accuracy_cnt += np.sum(p == t[i:i+batch_size]) #compare predict result to label
# end_time = time.perf_counter()   

# print("Accuracy : " + str(float(accuracy_cnt) / len(x)))
# print(f"time elapsed : {int(round((end_time - start_time) * 1000))}ms")

x_train, t_train, x_test, t_test = net.get_data_all()

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
print(f"batch_mask : {batch_mask}")
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(f"x_batch : {x_batch}")
print(f"t_batch : {t_batch}")