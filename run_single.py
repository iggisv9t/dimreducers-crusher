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
@click.option("--dataset-name", type=click.Choice(DATASET_REGISTRY))
@click.option("--metric-name", type=click.Choice(METRIC_REGISTRY), multiple=True)
@click.option("--reducer-name", type=click.Choice(REDUCER_REGISTRY))
def main(dataset_name, metric_name, reducer_name):
    print('---{}'.format(metric_name))
    now = time.strftime("%y%m%d_%H%M%S")
    report = dict()
    picsdir = "./pics"
    repdir = "./reports"
    print("====================")
    print("=====DIMCRUSHER=====")
    print("====================")

    datagen = DATASET_REGISTRY[dataset_name]()
    data = datagen.get(n=10000, d=10)
    print(data.shape, data.min(), data.max())

    reducer = REDUCER_REGISTRY[reducer_name]()
    data_reduced = reducer.fit_transform(data)
    print(data_reduced.shape, data_reduced.min(), data_reduced.max())

    # TODO: Allow multiple metrics
    for mn in metric_name:
        metric = METRIC_REGISTRY[mn]()
        metric_value = metric.score(data, data_reduced)
        print(metric_value)
        metric_report = {mn: metric_value}

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
