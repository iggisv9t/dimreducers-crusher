from .AbstractReducer import AbstractReducer
from sklearn.decomposition import IncrementalPCA as skipca
import numpy as np
import warnings


class IPCA(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        super().__init__(d, random_state)
        warnings.warn("Setting random seed does not affect IPCA.", UserWarning)
        self._main = skipca(n_components=d, **kwargs)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.fit_transform(x)

    def fit(self, x: np.ndarray, **kwargs):
        return self._main.fit(x)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.transform(x)

    def set_random_state(self, random_state: int = 0):
        warnings.warn("Setting random seed does not affect IPCA.", UserWarning)

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return True

    @staticmethod
    def get_parameter_ranges() -> dict:
        return {'whiten': (bool, False), 'batch_size': (int, 10, 100)}

