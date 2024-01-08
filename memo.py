import time
import polars as po
import matplotlib.pyplot as plt

cupy_enable = True
try:
    import cupy as np
except ImportError:
    import numpy as np
    cupy_enable = False

A = np.array([1, 2, 3])

print(A)