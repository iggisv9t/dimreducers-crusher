from .AbstractDatasetGenerator import AbstractDatasetGenerator
from typing import Union
import numpy as np
import scipy.sparse as sps


class RandomUniformDatasetGenerator(AbstractDatasetGenerator):

    def __init__(self, random_state: int = 0):
        super().__init__(random_state)

    def get(self, n: int, d: int, low: int = 0, high: int = 1, **kwargs) -> Union[sps.spmatrix, np.ndarray]:
        np_state = np.random.get_state()  # TODO: abstract this hack away to a decorator?
        np.random.seed(self.random_state)
        try:
            return np.random.uniform(low, high, (n, d))
        finally:
            np.random.set_state(np_state)

    @property
    def is_sparse(self) -> bool:
        return False

    def set_random_state(self, random_state: int = 0):
        self.random_state = random_state
