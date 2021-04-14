from abc import ABC, abstractmethod


class AbstractPlotter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def plot(x, y, colors=None, labels=None):
        pass
