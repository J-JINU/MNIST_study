# import numpy as np
cupy_enable = True
try:
    import cupy as np
except ImportError:
    import numpy as np
    cupy_enable = False


def init_network():
    pass

def forward(network, x):
    pass

