# import os
# import time
# import polars as po
# import matplotlib.pyplot as plt
# from modules import neuralnet as nnet
# from modules import function as func
# from PIL import Image

# import cupy as np
# # cupy_enable = True
# # try:
# #     import cupy as np
# # except ImportError:
# #     import numpy as np
# #     cupy_enable = False

# from deep_study.dataset.mnist import load_mnist

# # arr = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])
# # print(arr.shape)

# arr = np.arange(24).reshape(2,3,4)
# print(arr.shape)
# print(arr)

# arr2 = np.arange(24).reshape(2,4,3)
# print(arr2.shape)
# print(arr2)

# np.matmul(arr, arr2)
# print(np.matmul(arr, arr2))
# print(np.matmul(arr, arr2).shape)

# original_shape = arr.shape
# arr = arr.reshape(arr.shape[0], -1)
# print(arr.shape)
# print(arr)

# original_shape2 = arr2.shape
# arr2 = arr2.reshape(arr2.shape[0], -1)
# print(arr2.shape)
# print(arr2)

# print(np.matmul(arr, arr2.T))
# print(np.matmul(arr, arr2.T).shape)

# # (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# # x_train = np.asarray(x_train)
# # t_train = np.asarray(t_train)
# # x_test = np.asarray(x_test)
# # t_test = np.asarray(t_test)

# # train_loss_list = []

# # iters_num = 10000
# # train_size = x_train.shape[0]
# # batch_size = 100
# # learning_rate = 0.1
# # network = nnet.TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# # pred_start = time.perf_counter()
# # for i in range(iters_num):
# #     print(f'start {i} iter')
# #     start_time = time.perf_counter()
# #     batch_mask = np.random.choice(train_size, batch_size)
# #     x_batch = x_train[batch_mask]
# #     t_batch = t_train[batch_mask]
    
# #     grad = network.numerical_gradient(x_batch, t_batch)
    
# #     for key in ('W1', 'b1', 'W2', 'b2'):
# #         network.params[key] -= learning_rate * grad[key]
        
# #     loss = network.loss(x_batch, t_batch)
# #     train_loss_list.append(loss)
# #     end_time = time.perf_counter()
# #     print(f'end {i} iter, run time : {end_time - start_time}')
    
# # pred_end = time.perf_counter()
# # print(f'end pred, total time : {pred_end - pred_start}')
# # # 그래프 그리기
# # markers = {'train': 'o', 'test': 's'}
# # x = np.arange(len(train_loss_list))
# # plt.plot(x, train_loss_list, label='train loss')
# # plt.xlabel("epochs")
# # plt.ylabel("accuracy")
# # plt.ylim(0, 1.0)
# # plt.legend(loc='lower right')
# # plt.show()



# # cupy numpy compare time
# import cupy as cp
# import numpy as np
# import time as tp

# A      = np.random.rand(9000,9000) # NumPy rand
# G      = cp.random.rand(9000,9000) # CuPy rand
# G32    = cp.random.rand(3000,3000,dtype=cp.float32) # Create float32 matrix instead of float64 (default)
# G32_9k = cp.random.rand(9000,1000,dtype=cp.float32) # Create float32 matrix of a different shape

# t1 = tp.time()
# np.linalg.svd(A) # NumPy Singular Value Decomposition
# t2 = tp.time()
# print("CPU time: ", t2-t1)

# t3 = tp.time()
# cp.linalg.svd(G) # CuPy Singular Value Decomposition
# cp.cuda.Stream.null.synchronize() # Waits for GPU to finish
# t4 = tp.time()
# print("GPU time: ", t4-t3)

# t5 = tp.time()
# cp.linalg.svd(G32)
# cp.cuda.Stream.null.synchronize()
# t6 = tp.time()
# print("GPU float32 time: ", t6-t5)

# t7 = tp.time()
# cp.linalg.svd(G32_9k)
# cp.cuda.Stream.null.synchronize()
# t8 = tp.time()
# print("GPU float32 restructured time: ", t8-t7)