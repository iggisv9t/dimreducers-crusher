from .AbstractReducer import AbstractReducer
import numpy as np


class NCVis(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        import ncvis
        super().__init__(d, random_state)
        self._main = ncvis.NCVis(d=d, random_seed=random_state, **kwargs)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.fit_transform(x)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, **kwargs):
        raise NotImplementedError
    
    def set_random_state(self, random_state: int = 0):
        raise NotImplementedError

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_stateful(self) -> bool:
        return False

    @staticmethod
    def get_parameter_ranges() -> dict:
        return {
            'n_neighbors': (int, 2, 300),
            'min_dist': (float, 0, 1)
        }
