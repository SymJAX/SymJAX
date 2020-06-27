#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import symjax

x = symjax.tensor.Placeholder((0, 2), "float32")
w = symjax.tensor.Variable(1.0, dtype="float32")
p = x.sum(1)
f = symjax.function(x, outputs=p, updates={w: x.sum()})

print(f(np.ones((1, 2))))
print(w.value)
print(f(np.ones((2, 2))))
print(w.value)
# [2.]
# 2.0
# [2. 2.]
# 4.0

x = symjax.tensor.Placeholder((0, 2), "float32")
y = symjax.tensor.Placeholder((0,), "float32")
w = symjax.tensor.Variable((1, 1), dtype="float32")

loss = ((x.dot(w) - y) ** 2).mean()

g = symjax.gradients(loss, [w])[0]

other_g = symjax.gradients(x.dot(w).sum(), [w])[0]
f = symjax.function(x, y, outputs=loss, updates={w: w - 0.1 * g})
other_f = symjax.function(x, outputs=other_g)
for i in range(10):
    print(f(np.ones((i + 1, 2)), -1 * np.ones(i + 1)))
    print(other_f(np.ones((i + 1, 2))))

# 9.0
# [1. 1.]
# 3.2399998
# [2. 2.]
# 1.1663998
# [3. 3.]
# 0.419904
# [4. 4.]
# 0.15116541
# [5. 5.]
# 0.05441956
# [6. 6.]
# 0.019591037
# [7. 7.]
# 0.007052775
# [8. 8.]
# 0.0025389965
# [9. 9.]
# 0.0009140394
# [10. 10.]
