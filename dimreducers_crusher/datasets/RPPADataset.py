from .AbstractDataset import AbstractDataset
from typing import Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import scipy.sparse as sps
import warnings

# Original datasource: https://gdc.cancer.gov/about-data/publications/pancanatlas
# RNA sequencing data to classify 31 tumour types
# Many NA values!
# Shape: (7790, 198)

class RPPADataset(AbstractDataset):

    def __init__(self):
        super().__init__()

    def get(self, fillna: bool = False, **kwargs) -> Union[sps.spmatrix, np.ndarray]:
        try:
            print("Downloading RPPA...")
            data = pd.read_csv("http://api.gdc.cancer.gov/data/fcbb373e-28d4-4818-92f3-601ede3da5e1", sep = '\t')
            data = data.iloc[:, 2:]
            if fillna:
                imp = SimpleImputer(strategy = 'constant', fill_value = 0)
                return imp.fit_transform(np.array(data))
            else:
                return np.array(data)
        except:
            raise RuntimeError

    @property
    def is_sparse(self) -> bool:
        return False