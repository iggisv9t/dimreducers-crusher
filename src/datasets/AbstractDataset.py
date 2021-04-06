from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import scipy.sparse as sps


class AbstractDataset(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get(self, **kwargs) -> Union[sps.spmatrix, np.ndarray]:
        pass

    @property
    @abstractmethod
    def is_sparse(self) -> bool:
        return False
