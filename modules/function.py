import polars as po
import matplotlib.pyplot as plt

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
    
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = -0.2
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


class activate:
    def __init__(self) -> None:
        pass
    
    
if __name__ == '__main__':
    import time
    start_time = time.time()
    
    x = np.arange(0, 10, 0.01)
    y1 = np.sin(x)
    y2 = np.cos(x)
    if cupy_enable:
        x = np.asnumpy(x)
        y1 = np.asnumpy(y1)
        y2 = np.asnumpy(y2)

    plt.plot(x, y1, label='sin')
    plt.plot(x, y2, label='cos')
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("프로그램 실행 시간:", execution_time, "초")
    
    plt.legend()
    plt.show()