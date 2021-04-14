import click
from dimreducers_crusher import datasets, metrics, reducers
from dimreducers_crusher.utils.py_utils import get_registry


DATASET_REGISTRY = get_registry(datasets)
METRIC_REGISTRY = get_registry(metrics, exclude_substr=[])
REDUCER_REGISTRY = get_registry(reducers)


@click.command()
@click.argument('dataset_name', type=click.Choice(DATASET_REGISTRY))
@click.argument('metric_name', type=click.Choice(METRIC_REGISTRY))
@click.argument('reducer_name', type=click.Choice(REDUCER_REGISTRY))
def main(dataset_name, metric_name, reducer_name):
    print('====================')
    print('=====DIMCRUSHER=====')
    print('====================')

    datagen = DATASET_REGISTRY[dataset_name]()
    data = datagen.get(1000, 10)
    print(data.shape, data.min(), data.max())

    reducer = REDUCER_REGISTRY[reducer_name]()
    data_reduced = reducer.fit_transform(data)
    print(data_reduced.shape, data_reduced.min(), data_reduced.max())

    metric = METRIC_REGISTRY[metric_name]()
    metric_value = metric.score(data, data_reduced)
    print(metric_value)


if __name__ == '__main__':
    main()
