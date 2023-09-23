import numpy as np
from numpy import ndarray

def rand_uniform_sinplex(dim:int, num:int | None=None) -> ndarray:
    if num is None:
        r = np.random.exponential(scale=1.0, size=dim)
        return r / sum(r)
    else:
        r = np.random.exponential(scale=1.0, size=dim * num).reshape((dim, num))
        return r / r.sum(axis=0)


def rand_unform_cube(dim:int, num: int | None = None) -> ndarray:
    if num is None:
        return np.random.random(dim)
    else:
        return np.random.random(dim * num).reshape((dim, num))


def calc_stat_dist(A:ndarray) -> ndarray:
    dim = A.shape[0]

    c = A.T - np.identity(dim)
    c = np.delete(c, dim - 1, axis=0)
    c = np.vstack([c, np.ones((1, dim))])
    v = np.zeros((dim,))
    v[-1] = 1.0
    return np.linalg.solve(c, v)
