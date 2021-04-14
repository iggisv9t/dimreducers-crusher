from .AbstractDataset import AbstractDataset
from typing import Union
import numpy as np
import scipy.sparse as sps
import torchvision
import warnings

# Original datasource: https://github.com/zalandoresearch/fashion-mnist
# Shape: (60000, 784)

class FashionMNISTDataset(AbstractDataset):

    def __init__(self):
        super().__init__()

    def get(self, data_dir: str, **kwargs) -> Union[sps.spmatrix, np.ndarray]:
        try:
            train = torchvision.datasets.FashionMNIST(data_dir, download = True)
            X = train.data.numpy().reshape(-1, 28 * 28)
            return sps.csr_matrix(X)
        except:
            raise RuntimeError

    @property
    def is_sparse(self) -> bool:
        return True