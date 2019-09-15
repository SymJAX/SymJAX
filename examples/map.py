import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import theanoxla.nn as nn



w = T.Placeholder((3,), np.float32, name='w')

output = T.map(lambda a, b: T.pow(a, b), (w, T.cast(T.arange(3), 'float32')))
print(output.get({w: jax.numpy.arange(3).astype('float32')}))
fn = theanoxla.function(w, outputs=[output])
print(fn(jax.numpy.arange(3).astype('float32')))

output = T.scan(lambda a, b: a + b, T.zeros(1), T.reshape(w, (3, 1)))
print(output.get({w: jax.numpy.arange(3).astype('float32')}))
fn = theanoxla.function(w, outputs=[output])
print(fn(jax.numpy.arange(3).astype('float32')))

value = T.Placeholder((), np.float32)
output = T.cond(value < 0, (value, w), lambda a, b: a * b, (value, w),
                lambda a, b: a*b)
print(output.get({value: -1., w: jax.numpy.arange(3).astype('float32')}))
print(output.get({value: 1., w: jax.numpy.arange(3).astype('float32')}))
fn = theanoxla.function(value, w, outputs=[output])
print(fn(-1., jax.numpy.arange(3).astype('float32')))
print(fn(1., jax.numpy.arange(3).astype('float32')))


exit()




exit()

w = T.Placeholder((3,), 'float32', name='w')

output = T.map(lambda i:w*i, T.arange(4))

fn = theanoxla.function(w, outputs=[output])

print(fn(jax.numpy.ones(3)))
exit()
# user gives
def v1(x):
    return T.sum(T.square(x))

# we transform into
def theanofn_to_jaxfn(fn, *args, **kwargs):
    # treat the args
    pargs = list()
    for arg in args:
        pargs.append(T.placeholder_like(arg))
    # treat the kwargs
    pkwargs = dict()
    for name, var in kwargs.items():
        pkwargs[name] = T.placeholder_like(var)
    output = fn(*pargs, **pkwargs)
    def func(*fnargs, **fnkwargs):
        # treat the args
        pargs = list()
        for arg in fnargs:
            pargs.append(T.placeholder_like(arg))
        # treat the kwargs
        pkwargs = dict()
        for name, var in fnkwargs.items():
            pkwargs[name] = T.placeholder_like(var)
        feed_dict = zip(pargs, fnargs) +\
                    zip(pkwargs.values(), fnkwargs.values())
        return output.get(dict(feed_dict))

    return func



# we transform into
def _theanofn_to_jaxfn(*args, _fn, **kwargs):
    # treat the args
    pargs = list()
    for arg in args:
        pargs.append(T.placeholder_like(arg))
    # treat the kwargs
    pkwargs = dict()
    for name, var in kwargs.items():
        pkwargs[name] = T.placeholder_like(var)
    output = fn(*pargs, **pkwargs)
    feed_dict = zip(pargs, args) + zip(pkwargs.values(), kwargs.values())
    return output.get(dict(feed_dict))



print(v2(jax.numpy.ones((10, 10))))
exit()
w = T.Placeholder((3,), 'float32', name='w')

output = T.map(lambda i:w*i, T.range(4))

fn = theanoxla.function(w, outputs=[output])

print(fn(jax.numpy.ones(3)))
