import sys
sys.path.insert(0, "../")

import symjax as sj
import symjax.tensor as T
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

###### 2D GAUSSIAN EXAMPLE

t = T.linspace(-5, 5, 5)
x, y = T.meshgrid(t, t)
X = T.stack([x.flatten(), y.flatten()], 1)
p = T.pdfs.multivariate_normal.pdf(X, T.zeros(2), T.eye(2))
p = p.reshape((5, 5)).round(2)

print(p)
# Tensor(Op=round_, shape=(5, 5), dtype=float32)

# lazy evaluation (not compiled nor optimized)
print(p.get())
# [[0.   0.   0.   0.   0.  ]
#  [0.   0.   0.01 0.   0.  ]
#  [0.   0.01 0.16 0.01 0.  ]
#  [0.   0.   0.01 0.   0.  ]
#  [0.   0.   0.   0.   0.  ]]

# create the function which internall compiles and optimizes
# the function does not take any arguments and only outputs the
# computed tensor p
f = sj.function(outputs=p)
print(f())
# [[0.   0.   0.   0.   0.  ]
#  [0.   0.   0.01 0.   0.  ]
#  [0.   0.01 0.16 0.01 0.  ]
#  [0.   0.   0.01 0.   0.  ]
#  [0.   0.   0.   0.   0.  ]]


