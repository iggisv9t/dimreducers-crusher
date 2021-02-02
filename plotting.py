import matplotlib.pyplot as plt


def plot_2d_density(data, savepath=None, **kwargs):
    fig = plt.figure(figsize=(10, 10))
    plt.histogram2d(data[:, 0], data[:, 1], bins=200, **kwargs)

    if not (savepath is None):
        plt.savefig(savepath)

    return fig
