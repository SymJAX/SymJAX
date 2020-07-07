"""
Batch-Normalization example
===========================

example of batch-normalization classification
"""


import symjax.tensor as T
import symjax as sj
from symjax import nn
import numpy as np


def test_bn():
    sj.current_graph().reset()
    BATCH_SIZE = 5
    DIM = 2
    input = T.Placeholder((BATCH_SIZE, DIM), "float32", name="input")
    deterministic = T.Placeholder((1,), "bool", name="deterministic")

    bn = nn.layers.BatchNormalization(input, [1], deterministic=deterministic)

    update = sj.function(input, deterministic, outputs=bn, updates=sj.get_updates())
    get_stats = sj.function(input, outputs=bn.avg_mean)

    data = np.random.randn(50, DIM) * 4 + 2

    true_means = []
    actual_means = []

    for i in range(10):
        batch = data[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]
        output = update(batch, 0)
        assert np.allclose(
            output, (batch - batch.mean(0)) / (1e-4 + batch.std(0)), 1e-4
        )
        actual_means.append(get_stats(batch))
        if i == 0:
            true_means.append(batch.mean(0))
        else:
            true_means.append(0.9 * true_means[-1] + 0.1 * batch.mean(0))

    true_means = np.array(true_means)
    actual_means = np.array(actual_means).squeeze()

    assert np.allclose(true_means, actual_means, 1e-4)


test_bn()
