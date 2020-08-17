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
    np.random.seed(0)
    sj.current_graph().reset()
    BATCH_SIZE = 5
    DIM = 2
    input = T.Placeholder((BATCH_SIZE, DIM), "float32", name="input")
    deterministic = T.Placeholder((1,), "bool", name="deterministic")

    bn = nn.layers.BatchNormalization(input, [1], deterministic=deterministic)

    update = sj.function(input, deterministic, outputs=bn, updates=sj.get_updates())
    get_stats = sj.function(input, outputs=bn.avg_mean[0])

    data = np.random.randn(50, DIM) * 4 + 2

    true_means = [np.zeros(DIM)]
    actual_means = [np.zeros(DIM)]

    for i in range(10):
        batch = data[BATCH_SIZE * i : BATCH_SIZE * (i + 1)]

        output = update(batch, 0)
        assert np.allclose(
            output, (batch - batch.mean(0)) / np.sqrt(0.001 + batch.var(0)), 1e-4,
        )

        actual_means.append(get_stats(batch))
        true_means.append(0.99 * true_means[-1] + 0.01 * batch.mean(0))

    true_means = np.array(true_means)
    actual_means = np.array(actual_means).squeeze()
    assert np.allclose(true_means, actual_means)


def test_dropout():
    np.random.seed(0)
    sj.current_graph().reset()
    BATCH_SIZE = 4096
    DIM = 8
    input = T.Placeholder((BATCH_SIZE, DIM), "float32", name="input")
    deterministic = T.Placeholder((1,), "bool", name="deterministic")

    bn = nn.layers.Dropout(input, p=0.2, deterministic=deterministic)

    update = sj.function(input, deterministic, outputs=bn)

    data = np.ones((BATCH_SIZE, DIM))

    output1 = update(data, 0)
    output2 = update(data, 0)
    output3 = update(data, 1)

    assert not np.allclose(output1, output2, 1e-1)
    assert np.allclose(output1.mean(0) / 2 + output2.mean(0) / 2, 0.2, 0.08)
    assert np.all(output3)


def test_global_pool():
    np.random.seed(0)
    sj.current_graph().reset()
    BATCH_SIZE = 4096
    DIM = 8
    input = T.Placeholder((BATCH_SIZE, DIM), "float32", name="input")

    output = nn.layers.Dense(input, 64)
    output = nn.layers.Dense(output, output.shape[-1] * 2)
    output = nn.layers.Dense(output, output.shape[-1] * 2)
    get = sj.function(input, outputs=output)
    assert get(np.ones((BATCH_SIZE, DIM))).shape == (BATCH_SIZE, 64 * 4)


def test_flip():
    np.random.seed(0)
    sj.current_graph().reset()
    BATCH_SIZE = 2048
    DIM = 8
    input = T.Placeholder((BATCH_SIZE, DIM, DIM), "float32", name="input")
    deterministic = T.Placeholder((1,), "bool", name="deterministic")

    bn = nn.layers.RandomFlip(input, axis=2, p=0.5, deterministic=deterministic)

    update = sj.function(input, deterministic, outputs=bn)

    data = np.ones((BATCH_SIZE, DIM, DIM))
    data[:, :, : DIM // 2] = 0

    output1 = update(data, 0)
    output2 = update(data, 0)
    output3 = update(data, 1)

    assert not np.allclose(output1, output2, 1e-1)
    assert np.allclose(output1.mean(0) / 2 + output2.mean(0) / 2, 0.5, 0.05)
    assert np.allclose(data, output3, 1e-6)


if __name__ == "__main__":

    # test_bn()
    # test_flip()
    # test_dropout()
    test_global_pool()
