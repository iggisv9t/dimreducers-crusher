class BaseReducer:
    def __init__(self, outdim=2, **kwargs):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(
        self, X,
    ):
        self.fit(X)
        return self.transform(X)


class UmapReducer(BaseReducer):
    def __init__(self, outdim=2, **kwargs):
        import umap

        self.reducer = umap.UMAP(n_components=outdim, **kwargs)

    def fit(self, X):
        self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)


class TrimapReducer(BaseReducer):
    def __init__(self, outdim=2, **kwargs):
        import trimap

        self.reducer = trimap.TRIMAP(n_dims=outdim)

    def fit(self, X):
        self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.embedding_


class PyMDENeigh(BaseReducer):
    def __init__(self, outdim=2, **kwargs):
        self.outdim = outdim
        self.kwargs = kwargs

    def fit(self, X):
        import pymde

        self.reducer = pymde.preserve_neighbors(
            X, embedding_dim=self.outdim, **self.kwargs
        )

    def transform(self, X):
        return self.reducer.embed()
