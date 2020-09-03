import gc

import numpy as np

from assembly_calculus.brain.performance.multithreaded import multithreaded
from assembly_calculus.brain.performance.random_matrix import RandomMatrix


"""
Multithreaded tests
-------------------
These tests check the multithreaded decorator, using a simple usage of them.
"""

gc.collect()


@multithreaded
def sum_list(list_chunk):
    return sum(list_chunk)


@sum_list.set_params
def sum_list_params(thread_count, lst):
    n = int(np.ceil(len(lst) / thread_count))
    return [((lst[n * i:n * i + n],), {}) for i in range(thread_count)]


@sum_list.set_after
def sum_list_after(sums):
    return sum([s or 0 for s in sums])


def test_sum_list():
    assert sum_list(list(range(1000))) == sum(range(1000))


"""
RandomMatrix tests
------------------
These tests check the RandomMatrix class. They are statistical tests to check that the 
random generation works correctly.
"""

N = 1000
P = 0.4
# Statistical significance of 1%
EPSILON = 0.01


def expectation(a):
    return np.sum(a) / a.size


def cov(a, b):
    return expectation(a * b) - expectation(a) * expectation(b)


def test_expectation():
    arr = RandomMatrix().multi_generate(N, N, P)
    assert abs(expectation(arr) - P) < EPSILON


# Check that the different random generators used by different threads are uncorrelated.
def test_correlation():
    arr = RandomMatrix().multi_generate(N, N, P)
    assert cov(arr[0:400], arr[400:800]) < EPSILON
