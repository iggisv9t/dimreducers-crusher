from .AbstractMetric import AbstractMetric
from typing import Optional, Union
import numpy as np


class DummyMetric(AbstractMetric):
    def __init__(self, random_state: int = 0):
        super().__init__(random_state)

    def score(self, x: np.ndarray, x_reduced: np.ndarray,
              y: Optional[np.ndarray] = None, **kwargs) -> Union[float, np.ndarray]:
        return 1

    def set_random_state(self, random_state: int = 0):
        self.random_state = random_state

    @property
    def is_deterministic(self) -> bool:
        return True
