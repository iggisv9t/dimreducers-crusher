from .AbstractReducer import AbstractReducer
import numpy as np
import forcelayout as fl
import warnings


class ForcePivot(AbstractReducer):
    def __init__(self, d: int = 2, random_state: int = 0, **kwargs):
        super().__init__(d, random_state)
        warnings.warn("Setting random seed does not affect ForcePivot.", UserWarning)
        warnings.warn("ForcePivot supports only d = 2.", UserWarning)
        self.fitted = None

    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        self.fitted = fl.spring_layout(dataset = x, algorithm = fl.Pivot)
        return self.fitted.spring_layout()

    def fit(self, x: np.ndarray, **kwargs):
        raise NotImplementedError

    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def set_random_state(self, random_state: int = 0):
        warnings.warn("Setting random seed does not affect ForcePivot.", UserWarning)

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def is_stateful(self) -> bool:
        return True

    @staticmethod
    def get_parameter_ranges() -> dict:
        return None
