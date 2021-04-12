from abc import ABC, abstractmethod
import numpy as np


class AbstractReducer(ABC):

    def __init__(self, d: int = 2, random_state: int = 0):
        super().__init__()
        self.d = d
        self.random_state = random_state

    @abstractmethod
    def fit_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def set_random_state(self, random_state: int = 0):
        pass

    @property
    @abstractmethod
    def is_deterministic(self) -> bool:
        return False

    @property
    @abstractmethod
    def is_stateful(self) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def get_parameter_ranges() -> dict:
        return {}
