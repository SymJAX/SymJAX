import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

SHAPE = (4, 4)
a = T.Variable(np.random.randn(4, 4).astype('float32'))
q = a*2
#a = T.Variable(np.random.randn(4, 4).astype('float32'), name='a')
#b = T.Placeholder((4, 4), 'float32')
#b = T.Placeholder((4, 4), 'float32', name='b')
#c = T.ones((4, 4))
#print(c)
#c = T.ones((4, 4), name='c')
#print(c)
