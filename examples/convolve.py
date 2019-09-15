import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

def conv(x, y):
    x=x[0, 0]
    y=y[0, 0]
    filter = y[::-1, ::-1]
    return x[:-1,:-1]*y[0, 0]+x[1:,:-1]*y[1,0]+x[:-1,1:]*y[0,1]+x[1:,1:]*y[1, 1]

SHAPE = (1, 1, 1164, 1164)
w = T.Placeholder(SHAPE, 'float32', name='w')
SHAPE2 = (1, 1, 2, 2)
filter = T.Placeholder(SHAPE2, 'float32', name='filter')
output = T.convNd(w, filter)

f = theanoxla.function(w, filter, outputs=[output])

data = np.random.randn(*SHAPE).astype('float32')
filter = np.random.randn(*SHAPE2).astype('float32')
output = f(data, filter)[0][0, 0]
target = conv(data,filter)
print('% close values:', 100*np.mean(np.isclose(target,
                                                    output).astype('float32')))

