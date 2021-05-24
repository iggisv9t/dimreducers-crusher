from .AbstractReducer import AbstractReducer
from sklearn.decomposition import NMF as sknmf
import numpy as np


class NMF(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        super().__init__(d, random_state)
        self._main = sknmf(n_components = d, random_state = random_state, **kwargs)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if np.all(x > 0):
            return self._main.fit_transform(x)
        else:
            raise UserWarning('NMF works only for positive X.')

    def fit(self, x: np.ndarray, **kwargs):
        if np.all(x > 0):
            return self._main.fit(x)
        else:
            raise UserWarning('NMF works only for positive X.')

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        if np.all(x > 0):
            return self._main.transform(x)
        else:
            raise UserWarning('NMF works only for positive X.')

    def set_random_state(self, random_state: int = 0):
        self.random_state = random_state
        self._main.random_state = random_state

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_stateful(self) -> bool:
        return True

    @staticmethod
    def get_parameter_ranges() -> dict:
        return {'alpha': (int, 0, 10)}

