"""
Basic image transform (TPS/affine)
==================================

In this example we demonstrate how to employ the utility functions from
``symjax.tensor.interpolation.affine_transform`` and
``symjax.tensor.interpolation.thin_plate_spline``
to transform/interpolate images

"""

import matplotlib.pyplot as plt
import symjax
import symjax.tensor as T
import numpy as np

x = T.Placeholder((10, 1, 28, 28), "float32")
points = T.Placeholder((10, 2 * 16), "float32")
thetas = T.Placeholder((10, 6), "float32")

affine = T.interpolation.affine_transform(x, thetas)
tps = T.interpolation.thin_plate_spline(x, points)

f = symjax.function(x, thetas, outputs=affine)
g = symjax.function(x, points, outputs=tps)


data = symjax.data.mnist()["train_set/images"][:10]


plt.figure(figsize=(20, 6))
plt.subplot(2, 8, 1)
plt.imshow(data[0][0])
plt.title("original")
plt.ylabel("TPS")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 2)
points = np.zeros((10, 2 * 16))
plt.imshow(g(data, points)[0][0])
plt.title("identity")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 3)
points = np.zeros((10, 2 * 16))
points[:, :16] += 0.3
plt.imshow(g(data, points)[0][0])
plt.title("x translation")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 4)
points = np.zeros((10, 2 * 16))
points[:, 16:] += 0.3
plt.imshow(g(data, points)[0][0])
plt.title("y translation")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 5)
points = np.random.randn(10, 2 * 16) * 0.2
plt.imshow(g(data, points)[0][0])
plt.title("random")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 6)
points = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
points = np.concatenate([points[0].reshape(-1), points[1].reshape(-1)]) * 0.4
points = points[None] * np.ones((10, 1))
plt.imshow(g(data, points)[0][0])
plt.title("zoom")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 7)
points = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
points = np.concatenate([points[0].reshape(-1), points[1].reshape(-1)]) * -0.2
points = points[None] * np.ones((10, 1))
plt.imshow(g(data, points)[0][0])
plt.title("zoom")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 8)
points = np.zeros((10, 2 * 16))
points[:, 1::2] -= 0.1
points[:, ::2] += 0.1
plt.imshow(g(data, points)[0][0])
plt.title("blob")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 9)
plt.imshow(data[0][0])
plt.title("original")
plt.ylabel("Affine")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 10)
points = np.zeros((10, 6))
points[:, 0] = 1
points[:, 4] = 1
plt.imshow(f(data, points)[0][0])
plt.title("identity")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 11)
points = np.zeros((10, 6))
points[:, 0] = 1
points[:, 4] = 1
points[:, 2] = 0.2
plt.imshow(f(data, points)[0][0])
plt.title("x translation")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 12)
points = np.zeros((10, 6))
points[:, 0] = 1
points[:, 4] = 1
points[:, 5] = 0.2
plt.imshow(f(data, points)[0][0])
plt.title("y translation")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 13)
points = np.zeros((10, 6))
points[:, 0] = 1
points[:, 4] = 1
points[:, 1] = 0.4
plt.imshow(f(data, points)[0][0])
plt.title("skewness x")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 14)
points = np.zeros((10, 6))
points[:, 0] = 1.4
points[:, 4] = 1.4
plt.imshow(f(data, points)[0][0])
plt.title("zoom")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 15)
points = np.zeros((10, 6))
points[:, 0] = 1.4
points[:, 4] = 1.0
plt.imshow(f(data, points)[0][0])
plt.title("zoom x")
plt.xticks([])
plt.yticks([])

plt.subplot(2, 8, 16)
points = np.zeros((10, 6))
points[:, 0] = 1
points[:, 4] = 1
points[:, 3] = 0.4
plt.imshow(f(data, points)[0][0])
plt.title("skewness y")
plt.xticks([])
plt.yticks([])


plt.tight_layout()
plt.show()
