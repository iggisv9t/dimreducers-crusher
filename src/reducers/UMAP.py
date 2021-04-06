from .AbstractReducer import AbstractReducer
import numpy as np


class UMAP(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        import umap
        super().__init__(d, random_state)
        self._main = umap.UMAP(n_components=d, random_state=random_state, **kwargs)

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.fit_transform(x)

    def fit(self, x: np.ndarray, **kwargs):
        return self._main.fit(x)

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return self._main.transform(x)

    def set_random_state(self, random_state: int = 0):
        self.random_state = random_state
        self._main.random_state = random_state

    @property
    def is_deterministic(self) -> bool:
        return False

    @property
    def is_stateful(self) -> bool:
        return True
