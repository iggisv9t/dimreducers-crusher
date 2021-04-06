from .AbstractReducer import AbstractReducer
import numpy as np


class DummyReducer(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0):
        super().__init__(d, random_state)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return x[:, :self.d]

    def fit(self, x: np.ndarray, **kwargs):
        pass

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return x[:, :self.d]

    def set_random_state(self, random_state: int = 0):
        self.random_state = random_state

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return False

