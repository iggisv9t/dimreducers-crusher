from abc import abstractmethod
from typing import Union
from .AbstractDataset import AbstractDataset
import numpy as np
import scipy.sparse as sps


class AbstractDatasetGenerator(AbstractDataset):

    def __init__(self, random_state: int = 0):
        super().__init__()
        self.random_state = random_state

    @abstractmethod
    def get(self, n: int, d: int, **kwargs) -> Union[sps.spmatrix, np.ndarray]:
        pass

    @property
    @abstractmethod
    def is_sparse(self) -> bool:
        return False

    @abstractmethod
    def set_random_state(self, random_state: int = 0):
        pass
