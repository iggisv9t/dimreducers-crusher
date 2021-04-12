from .AbstractDataset import AbstractDataset
from typing import Union
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse as sps
import warnings

# Original datasource: http://qwone.com/~jason/20Newsgroups/
# Pipeline as in https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py
# Shape: (11314, 130107)

class TwentyNewsgroups(AbstractDataset):

    def __init__(self):
        super().__init__()

    def get(self, random_state: int = 42, **kwargs) -> Union[sps.spmatrix, np.ndarray]:
        try:
            dataset = fetch_20newsgroups(subset = 'train', random_state = random_state)
            cvec = CountVectorizer()
            tfidfvec = TfidfTransformer()
            data = cvec.fit_transform(dataset.data)
            data = tfidfvec.fit_transform(data)
            return data
        except:
            raise RuntimeError

    @property
    def is_sparse(self) -> bool:
        return True