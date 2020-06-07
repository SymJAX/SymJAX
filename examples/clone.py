import numpy as np
import symjax
import symjax.tensor as T

# we create a simple mapping with 2 matrix multiplications interleaved
# with nonlinearities
x = T.Placeholder((8,), 'float32')
w_1 = T.Variable(T.random.randn((16, 8)))
w_2 = T.Variable(T.random.randn((2, 16)))

# the output can be computed easily as
output = w_2.dot(T.relu(w_1.dot(x)))

# now suppose we also wanted the same mapping but with a noise input
epsilon = T.random.randn((8,))

output_noisy = output.clone({x:x+epsilon})

f = symjax.function(x, outputs=[output, output_noisy])

for i in range(10):
    print(f(np.ones(8)))

# [array([-14.496595,   8.7136  ], dtype=float32), array([-11.590391 ,   4.7543654], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-30.038504,  26.758451], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-19.214798,  19.600328], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-12.927457,  10.457445], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-19.486668,  17.367273], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-31.634314,  24.837488], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-19.756075,  12.330083], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-38.9738  ,  31.588022], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-19.561726,  12.192366], dtype=float32)]
# [array([-14.496595,   8.7136  ], dtype=float32), array([-33.110832,  30.104563], dtype=float32)]

