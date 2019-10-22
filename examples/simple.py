import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

SHAPE = (4, 4)
a = T.Variable(np.random.randn(4, 4).astype('float32'))
q = a*2
print(q[0].shape, '(4,)')
print(q[T.range(2)].shape, '(2, 4)')
print(q[[0,1]].shape, '(2, 4)')
#print(q[:, [0,1]].shape, '(4, 2)')

i = T.Variable(1)
print(T.dynamic_slice_in_dim(q, i, 1).get())
print('original dtype', q.dtype, q.get())
print('convert to', T.cast(q, 'float32').dtype, T.cast(q, 'float32').get())
print('convert to', T.cast(q, 'int32').dtype, T.cast(q, 'int32').get())




#a = T.Variable(np.random.randn(4, 4).astype('float32'), name='a')
#b = T.Placeholder((4, 4), 'float32')
#b = T.Placeholder((4, 4), 'float32', name='b')
#c = T.ones((4, 4))
#print(c)
#c = T.ones((4, 4), name='c')
#print(c)
