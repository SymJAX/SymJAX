"""
CIFAR10 classification
======================

example of image classification
"""
import symjax.tensor as T
import symjax as sj
from symjax import nn
import symjax
import numpy as np
import matplotlib.pyplot as plt
from symjax.data import mnist
from symjax.data.utils import batchify

import os

os.environ["DATASET_PATH"] = "/home/vrael/DATASETS/"

# load the dataset
mnist = mnist.load()

# some renormalization
mnist["train_set/images"] /= mnist["train_set/images"].max((1, 2, 3), keepdims=True)
mnist["test_set/images"] /= mnist["test_set/images"].max((1, 2, 3), keepdims=True)

# create the network
BATCH_SIZE = 32
images = T.Placeholder((BATCH_SIZE, 1, 28, 28), "float32", name="images")
labels = T.Placeholder((BATCH_SIZE,), "int32", name="labels")
deterministic = T.Placeholder((1,), "bool")


layer = [nn.layers.Identity(images)]

for l in range(3):
    layer.append(nn.layers.Conv2D(layer[-1], 32, (3, 3), b=None, pad="SAME"))
    layer.append(nn.layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(nn.layers.Lambda(layer[-1], nn.leaky_relu))
    layer.append(nn.layers.Pool2D(layer[-1], (2, 2)))

layer.append(nn.layers.Pool2D(layer[-1], layer[-1].shape[2:], pool_type="AVG"))
layer.append(nn.layers.Dense(layer[-1], 10))

# each layer is itself a tensor which represents its output and thus
# any tensor operation can be used on the layer instance, for example
for l in layer:
    print(l.shape)


loss = nn.losses.sparse_crossentropy_logits(labels, layer[-1]).mean()
accuracy = nn.losses.accuracy(labels, layer[-1])

lr = nn.schedules.PiecewiseConstant(0.01, {15: 0.001, 25: 0.0001})


nn.optimizers.Adam(loss, lr)

test = symjax.function(images, labels, deterministic, outputs=[loss, accuracy])

train = symjax.function(
    images,
    labels,
    deterministic,
    outputs=[loss, accuracy],
    updates=symjax.get_updates(),
)

test_accuracy = []

for epoch in range(3):
    L = list()
    for x, y in batchify(
        mnist["test_set/images"],
        mnist["test_set/labels"],
        batch_size=BATCH_SIZE,
        option="continuous",
    ):
        L.append(test(x, y, 1))
    print("Test Loss and Accu:", np.mean(L, 0))
    test_accuracy.append(np.mean(L, 0))
    L = list()
    for x, y in batchify(
        mnist["train_set/images"],
        mnist["train_set/labels"],
        batch_size=BATCH_SIZE,
        option="random_see_all",
    ):
        L.append(train(x, y, 0))
    print("Train Loss and Accu", np.mean(L, 0))
    lr.update()

plt.plot(test_accuracy)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.title("CIFAR10 classification task")
