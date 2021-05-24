from .AbstractReducer import AbstractReducer
from pydiffmap import diffusion_map as dm
import warnings
import numpy as np


class DiffusionMap(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        super().__init__(d, random_state)
        warnings.warn("Setting random seed does not affect DiffusionMap.", UserWarning)
        self._main = dm.DiffusionMap.from_sklearn(n_evecs = d)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.fit_transform(x)

    def fit(self, x: np.ndarray, **kwargs):
        return self._main.fit(x)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.transform(x)

    def set_random_state(self, random_state: int = 0):
        warnings.warn("Setting random seed does not affect DiffusionMap.", UserWarning)

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return True

    @staticmethod
    def get_parameter_ranges() -> dict:
        return {'alpha': (float, 0.1, 1.0),
                'k': (int, 20, 405)}

