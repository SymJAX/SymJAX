import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import symjax
import symjax.tensor as T

w = T.Placeholder((3,), 'float32', name='w')
w_interp1 = T.upsample_1d(w, repeat=4, axis=0, mode='nearest')
w_interp2 = T.upsample_1d(w, repeat=4, axis=0, mode='linear')
w_interp3 = T.upsample_1d(w, repeat=4, axis=0)



f = symjax.function(w, outputs=[w_interp1, w_interp2, w_interp3])


print(f([1,2,3]))
