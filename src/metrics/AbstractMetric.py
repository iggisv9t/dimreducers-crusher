from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class AbstractMetric(ABC):

    def __init__(self, random_state: int = 0):
        super().__init__()
        self.random_state = random_state

    @abstractmethod  # TODO: should this also be @staticmethod?
    def score(self, x: np.ndarray, x_reduced: np.ndarray,
              y: Optional[np.ndarray], distances: bool = False, **kwargs) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def set_random_state(self, random_state: int = 0):
        pass

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        return False
