"""
Batch-Normalization example
===========================

example of batch-normalization classification
"""


import symjax.tensor as T
import symjax as sj
from symjax import nn
import symjax
import numpy as np
import matplotlib.pyplot as plt
from symjax.data import mnist
from symjax.data.utils import batchify

v = T.Variable(False, trainable=False, dtype="bool")
f = symjax.function(outputs=v, updates={v: True})
print(f())
print(f())


BATCH_SIZE = 100
DIM = 2
input = T.Placeholder((BATCH_SIZE, DIM), "float32", name="input")
deterministic = T.Placeholder((1,), "bool")

# bn = nn.layers.BatchNormalization(input, [1], deterministic=deterministic)


manual_mean = nn.schedules.ExponentialMovingAverage(input.mean(0), 0.9)

v = T.Variable(
    False, trainable=False, name="first_step", dtype="bool"
)  # sj.get_variables("*first_step*")[0]
update = symjax.function(updates={v: True})  # updates=sj.get_updates())
get_stats = symjax.function(input, outputs=manual_mean)
# get_output = symjax.function(input, deterministic, outputs=bn)


get_step = symjax.function(outputs=v)

data = np.random.randn(1000, DIM) * 4 + 2

true_means = []
actual_means = []

for e in range(1):
    for i in range(3):
        #        batch = data[100 * i : 100 * (i + 1)]
        print("step", get_step())
        update()
        # actual_means.append(get_stats(batch)[1])
        # if i == 0:
        #     true_means.append(batch.mean(0))
        # else:
        #     true_means.append(0.9 * true_means[-1] + 0.1 * batch.mean(0))
sdf

true_means = np.array(true_means)
actual_means = np.array(actual_means)

print(true_means)
print(actual_means)
