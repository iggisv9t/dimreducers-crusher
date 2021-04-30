from .AbstractPlotter import AbstractPlotter
import matplotlib.pyplot as plt


class DefaultPlotter(AbstractPlotter):
    def __init__(self):
        super().__init__()

    def plot(self, x, savepath, y=None, colors=None, labels=None, figsize=(20, 20), **kwargs):
        fig = plt.figure(figsize=figsize)
        plt.hist2d(x[:, 0], x[:, 1], cmap="magma", bins=200)
        plt.savefig(savepath)
        return fig
