"""
Basic image resampling and alignment
====================================

demonstration on how to perform basic image preprocessing

"""

import matplotlib.pyplot as plt
import numpy as np
import symjax


image1 = np.random.rand(3, 2, 4)
image2 = np.random.rand(3, 4, 2)
image3 = np.random.rand(3, 4, 4)
all_images = [image1, image2, image3]

images = symjax.data.utils.resample_images(all_images, (6, 6))

fig = plt.figure(figsize=(8, 3))
for i in range(3):

    plt.subplot(2, 3, i + 1)
    plt.imshow(all_images[i].transpose(1, 2, 0), aspect="auto", vmax=10, cmap="jet")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 3, i + 4)
    plt.imshow(images[i].transpose(1, 2, 0), aspect="auto", vmax=10, cmap="jet")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
