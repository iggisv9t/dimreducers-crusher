import os
import time
import click
import json
from dimreducers_crusher import datasets, metrics, reducers, plotters
from dimreducers_crusher.utils.py_utils import get_registry


DATASET_REGISTRY = get_registry(datasets)
METRIC_REGISTRY = get_registry(metrics, exclude_substr=[])
REDUCER_REGISTRY = get_registry(reducers)


@click.command()
@click.argument("dataset_name", type=click.Choice(DATASET_REGISTRY))
@click.argument("metric_name", type=click.Choice(METRIC_REGISTRY))
@click.argument("reducer_name", type=click.Choice(REDUCER_REGISTRY))
def main(dataset_name, metric_name, reducer_name):
    now = time.strftime("%y%m%d_%H%M%S")
    report = dict()
    picsdir = "./pics"
    repdir = "./reports"
    print("====================")
    print("=====DIMCRUSHER=====")
    print("====================")

    datagen = DATASET_REGISTRY[dataset_name]()
    data = datagen.get(1000, 10)
    print(data.shape, data.min(), data.max())

    reducer = REDUCER_REGISTRY[reducer_name]()
    data_reduced = reducer.fit_transform(data)
    print(data_reduced.shape, data_reduced.min(), data_reduced.max())

    # TODO: Allow multiple metrics
    metric = METRIC_REGISTRY[metric_name]()
    metric_value = metric.score(data, data_reduced)
    print(metric_value)
    metric_report = {metric_name: metric_value}

    # Plotter
    # TODO: Allow multiple plotters
    fname = "{}_{}_{}".format(dataset_name, reducer_name, now)

    p = plotters.DefaultPlotter()
    os.makedirs(picsdir, exist_ok=True)

    picpath = os.path.join(picsdir, "{}.png".format(fname))
    p.plot(data_reduced, picpath)
    
    # Report
    os.makedirs(repdir, exist_ok=True)
    report["dataset"] = dataset_name
    report["reducer"] = reducer_name
    report["metrics"] = metric_report
    report["plots"] = [picpath]

    reppath = os.path.join(repdir, "{}.json".format(fname))
    with open(reppath, "w") as fp:
        json.dump(report, fp)


if __name__ == "__main__":
    main()
