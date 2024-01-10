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
x, t = net.get_data()
network = net.init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = net.predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))