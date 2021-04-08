import numpy as np
import scipy.sparse as sps
from typing import Union


def fast_euclidean(data: np.ndarray) -> np.ndarray:
    dd = np.sum(data * data, axis=1)
    dist = -2 * np.dot(data, data.T)
    dist += dd + dd[:, np.newaxis]
    np.fill_diagonal(dist, 0)
    np.sqrt(dist, dist)
    return dist


def distance_matrix(data: Union[np.ndarray, sps.spmatrix], metric: str = 'euclidean'):
    if metric in ('euclidean', 'euc') and isinstance(data, np.ndarray):
        return fast_euclidean(data)
    raise NotImplementedError

