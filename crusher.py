import numpy as np
import matplotlib.pyplot as plt


def crush_em_all(reducers, datasets, plotters, metrics, show=False):
    """
    reducers: list of reducer instances
    datasets: list of tuples of numpy arrays (X, y)
    plotters: functions to call on embeddings and labels to get pictures
    metrics
    """
    for reducer in reducers:
        for data, labels in datasets:
            # TODO: measure run time

            emb = reducer.fit_transform(data)

            for metric, name in metrics:
                metric(emb, labels)
                # TODO: print and save results

            # TODO: Different plotters
            for plotter in plotters:
                # TODO: save parameter and path generation
                plotter(emb, y=labels)

            # TODO: save logs
