import jax
import numpy as np
import sys

sys.path.insert(0, "../")

import symjax
import symjax.tensor as T


def mymean(x):
    return (x[::2, ::2] + x[1::2, ::2] + x[::2, 1::2] + x[1::2, 1::2]) / 4


def mysum(x):
    return mymean(x) * 4


def mymax(x):
    return np.maximum(
        np.maximum(x[::2, ::2], x[1::2, ::2]), np.maximum(x[::2, 1::2], x[1::2, 1::2])
    )


SHAPE = (4, 4)
w = T.Placeholder(SHAPE, "float32", name="w")
maxpool = T.poolNd(w, (2, 2), reducer="MAX")
meanpool = T.poolNd(w, (2, 2), reducer="AVG")
sumpool = T.poolNd(w, (2, 2), reducer="SUM")

f = symjax.function(w, outputs=[maxpool, meanpool, sumpool])

for i in range(10):
    data = np.random.randn(*SHAPE)
    output = f(data)
    print(
        np.allclose(mymean(data), output[1]),
        np.allclose(mymax(data), output[0]),
        np.allclose(mysum(data), output[2]),
    )
