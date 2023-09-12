import numpy as np
from numba_scipy_complex import gamma
from scipy.special import gamma as scipy_gamma
from numba import njit, prange
from time import time


@njit(parallel=True)
def numba():
    a = np.asarray([-0.6 + 0.3j, 12.4 - 3.1j, -4.4 - 1.4j, 3.7 + 21.1j])
    x = np.zeros(4, dtype=np.complex_)
    for i in prange(100):
        x += gamma(a) * i
    return x


def regular():
    a = np.asarray([-0.6 + 0.3j, 12.4 - 3.1j, -4.4 - 1.4j, 3.7 + 21.1j])
    x = np.zeros(4, dtype=np.complex_)
    for i in prange(100):
        x += scipy_gamma(a) * i
    return x


# compile
result = numba()

# measure
start = time()
for _ in range(1000):
    result = numba()
dur = time() - start
print(result, dur)
start = time()
for _ in range(1000):
    result = regular()
dur = time() - start
print(result, dur)
