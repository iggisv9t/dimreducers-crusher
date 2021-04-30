import sklearn
import numpy as np
from AbstractDataset import AbstractDataset 
from sklearn.datasets import fetch_covtype


class CovTypeDataset(AbstractDataset):

	def __init__(self):
		super().__init__()

	def get(self, **kwargs) -> np.ndarray:
		try:
			print('Fetching CovType dataset...')
			data = fetch_covtype(download_if_missing = True, shuffle = True)
			X, y = data['data'], data['target']

			return X, y

		except:
			raise RuntimeError

	@property
	def is_sparse(self) -> bool:
		return False