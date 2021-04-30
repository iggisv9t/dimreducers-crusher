import sklearn 
import numpy as np
from AbstractDataset import AbstractDataset
from sklearn.datasets import make_swiss_roll

# shape is defined by args in get
# default is (1000,3)

class SwissRollDataset(AbstractDataset):

	def __init__(self):
		super().__init__()

	def get(self,n_samples = 1000, noise = 0.1, **kwargs) -> np.ndarray:
		try:
			X, y = make_swiss_roll(n_samples = n_samples, noise = noise)
			return X,y
		except:
			RuntimeError

	@property
	def is_sparse(self) -> bool:
		return False
