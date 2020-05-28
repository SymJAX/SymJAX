import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import symjax
import symjax.tensor as T


# map
xx = T.ones(10)
a = T.map(lambda a: a * 2, xx)
g = symjax.gradients(a.sum(),xx)[0]
f = symjax.function(outputs=[a,g])

# scan
xx = T.ones(10) * 2
a = T.scan(lambda c, x: (c*x,c*x), T.ones(1), xx)
g = symjax.gradients(a[1][-1],xx)[0]
f = symjax.function(outputs=[a,g])

# scan with updates
xx = T.range(5)
uu = T.ones((10,2))
vvar = T.Variable(T.zeros((10,2)))
vv = T.index_add(vvar, 1, 1)
a = T.scan(lambda c, x, p: (T.index_update(c, x, p[x]),1), vv, xx, [vv])
#a = T.scan(lambda c, x: (c*x,c*x), T.ones(1), xx)
#a = T.scan(lambda c, x: (T.square(c),c[0]), uu, xx)
#g = symjax.gradients(a[1][-1],xx)
f = symjax.function(outputs=a[0], updates={vvar:vvar+1})
print(f(),f(), f())
asdf





# fori loop
b= T.Placeholder((), 'int32')
xx = T.ones(1)
a = T.fori_loop(0, b, lambda i,x:i*x, xx)
f = symjax.function(b,outputs=a)
print(f(0), f(1),f(2), f(3))


# COND example 1
value = T.Placeholder((), np.float32)
output = T.cond(value < 0, (value, w), lambda a, b: a * b, (value, w),
                lambda a, b: a*b)
print(output.get({value: -1., w: jax.numpy.arange(3).astype('float32')}))
print(output.get({value: 1., w: jax.numpy.arange(3).astype('float32')}))
fn = symjax.function(value, w, outputs=[output])
print(fn(-1., jax.numpy.arange(3).astype('float32')))
print(fn(1., jax.numpy.arange(3).astype('float32')))

# COND example 2
value = T.Placeholder((), np.float32)
output = T.cond(value < 0, value, lambda a: a * 10, value,
                lambda a: a * 20)
print(output.get({value: -1.}))
print(output.get({value: 1.}))
fn = symjax.function(value, outputs=[output])
print(fn(-1.))
print(fn(1.))








