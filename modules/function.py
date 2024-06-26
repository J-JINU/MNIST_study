import polars as po
import matplotlib.pyplot as plt

import numpy as np
cupy_enable = True
try:
    import cupy as np
except ImportError:
    import numpy as np
    cupy_enable = False

#gate function
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    꺤
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(x1, x2)

#activation function
def step_func(x):
    return np.array(x > 0, dtype=np.int8)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#output function
def softmax(a):
    # 주석의 방법으로 하면 오버플로가 발생할 수 있다. exponential함수는 발산하기 때문.
    # exp_a = np.exp(a)
    # return (exp_a / np.sum(exp_a)) 
    c = np.max(a)
    exp_a = np.exp(a - c)
    return (exp_a / np.sum(exp_a))

    """
    y : predict result
    t : real data
    """
def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    # delta = 1e-7 # log0 = -inf  -> lo g(y + delta) = not -inf
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # if one_hot is True:
    #     return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # else:
    #     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_diff(f, x):
    h = 1e-10
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient_1d(f, x): #just 1 dim
    h = 1e-10
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        
    return grad

def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_gradient_1d(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, X in enumerate(x):
            grad[idx] = numerical_gradient_1d(f, X)
    
    return grad

class activate:
    def __init__(self) -> None:
        pass
    
    
if __name__ == '__main__':
    # print(sigmoid(12312))
    # import time
    # start_time = time.time()
    
    # x = np.arange(0, 10, 0.01)
    # y1 = np.sin(x)
    # y2 = np.cos(x)
    # if cupy_enable:
    #     x = np.asnumpy(x)
    #     y1 = np.asnumpy(y1)
    #     y2 = np.asnumpy(y2)

    # plt.plot(x, y1, label='sin')
    # plt.plot(x, y2, label='cos')
    
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("프로그램 실행 시간:", execution_time, "초")
    
    # plt.legend()
    # plt.show()
    pass