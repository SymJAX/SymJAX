import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T



image = T.Placeholder((512**2,), 'float32')
output = image.reshape((1, 1, 512, 512))
f = theanoxla.function(image, outputs=[output])
for i in range(10000):
    print(i)
    f(np.random.randn(512**2))
