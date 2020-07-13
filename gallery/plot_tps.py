"""
Pixel interpolation learning
============================

we demonstrate in this toy example how to use the coordinate
interpolation techniques with learnable parameter to
map one image to another one simply by interpolation the original
image values from learned coordinates

"""

import symjax
import symjax.tensor as T
import matplotlib.pyplot as plt
import numpy as np

import os

os.environ["DATASET_PATH"] = "/home/vrael/DATASETS/"

symjax.current_graph().reset()


mnist = symjax.data.mnist.load()
# 2d image
images = mnist["train_set/images"][mnist["train_set/labels"] == 2][:2, 0]
images /= images.max()

np.random.seed(0)

coordinates = T.meshgrid(T.range(28), T.range(28))
coordinates = T.Variable(
    T.stack([coordinates[1].flatten(), coordinates[0].flatten()]).astype("float32")
)
interp = T.interpolation.map_coordinates(images[0], coordinates, order=1).reshape(
    (28, 28)
)

loss = ((interp - images[1]) ** 2).mean()

lr = symjax.nn.schedules.PiecewiseConstant(0.05, {5000: 0.01, 8000: 0.005})
symjax.nn.optimizers.Adam(loss, lr)

train = symjax.function(outputs=loss, updates=symjax.get_updates())

rec = symjax.function(outputs=interp)

losses = list()

original = coordinates.value

for i in range(100):
    losses.append(train())

reconstruction = rec()

after = coordinates.value


plt.figure(figsize=(12, 6))

plt.subplot(311)
plt.semilogy(losses, "-x")
plt.ylabel("loss (l2)")
plt.title("Training loss")


plt.subplot(334)
plt.imshow(images[0], aspect="auto", cmap="plasma")
plt.xticks([])
plt.yticks([])
plt.title("input")

plt.subplot(335)
plt.imshow(images[1], aspect="auto", cmap="plasma")
plt.xticks([])
plt.yticks([])
plt.title("target")

plt.subplot(336)
plt.imshow(reconstruction, aspect="auto", cmap="plasma")
plt.xticks([])
plt.yticks([])
plt.title("reconstruction")


print(original)

plt.subplot(325)
plt.scatter(original[1][::-1], original[0], s=3)
plt.xticks([])
plt.yticks([])
plt.title("Initialized coordinates")

plt.subplot(326)
plt.scatter(after[1][::-1], after[0], s=3)
plt.xticks([])
plt.yticks([])
plt.title("Learned coordinates")


plt.tight_layout()
plt.show()
