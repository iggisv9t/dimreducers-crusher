from .AbstractReducer import AbstractReducer
import numpy as np
import warnings
from sklearn.manifold import Isomap as skisomap


class Isomap(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        super().__init__(d, random_state)
        warnings.warn("Setting random seed does not affect Isomap.", UserWarning)
        self._main = skisomap(n_components = d, **kwargs)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.fit_transform(x)

    def fit(self, x: np.ndarray, **kwargs):
        return self._main.fit(x)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.transform(x)

    def set_random_state(self, random_state: int = 0):
        warnings.warn("Setting random seed does not affect Isomap.", UserWarning)

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return True

    @staticmethod
    def get_parameter_ranges() -> dict:
        return {
            'n_neighbors': (int, 2, 300),
            'p': (int, 1, 2)
        }
