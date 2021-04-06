import src.metrics, src.datasets, src.reducers

datagen = src.datasets.RandomUniformDatasetGenerator()
data = datagen.get(1000, 10, low=0, high=10)
print(data.shape, data.min(), data.max())

reducer = src.reducers.PCA()
data_reduced = reducer.fit_transform(data)
print(data_reduced.shape, data_reduced.min(), data_reduced.max())

metric = src.metrics.DummyMetric()
metric_value = metric.score(data, data_reduced)
print(metric_value)
