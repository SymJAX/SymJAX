
![SymJAX logo](./docs/img/symjax_logo.png)


# SymJAX: symbolic CPU/GPU/TPU programming ![Continuous integration](https://github.com/SymJAX/SymJAX/workflows/Continuous%20integration/badge.svg) ![doctest](https://github.com/SymJAX/SymJAX/workflows/doc/badge.svg) ![license](https://img.shields.io/badge/license-Apache%202-blue) <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
This is an under-development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs.
[**Reference docs**](https://symjax.readthedocs.io/en/latest/)


## What is SymJAX ?

SymJAX is a symbolic programming version of JAX simplifying graph input/output/updates and providing additional functionalities for general machine learning and deep learning applications. From an user perspective SymJAX apparents to Theano with fast graph optimization/compilation and broad hardware support, along with Lasagne-like deep learning functionalities

## Why SymJAX ?

The number of libraries topping Jax/Tensorflow/Torch is large and growing by the
day. What SymJAX offers as opposed to most is an all-in-one library with diverse
functionalities such as

- dozens of various datasets with clear descriptions and one line import
- versatile set of functions from ffts, linear algebraic tools, random variables, ...
- advanced signal processing tools such as multiple wavelet families (in time and frequency domain), multiple time-frequency representations, apodization windows, ...
- IO utilities to monitor/save/track specific statistics during graph execution through h5 files and numpy, simple and explicit graph saving allowing to save and load models without burden
- side utilities such as automatic batching of dataset, data splitting, cross-validation, ...

and most importantly, a SYMBOLIC/DECLARATIVE programming environment allowing CONCISE/EXPLICIT/OPTIMIZED computations.

For a deep network oriented imperative library built on JAX and with a JAX syntax check out [FLAX](https://github.com/google/flax).

## Examples

```python
import sys
import symjax as sj
import symjax.tensor as T

# create our variable to be optimized
mu = T.Variable(T.random.normal((), seed=1))

# create our cost
cost = T.exp(-(mu-1)**2)

# get the gradient, notice that it is itself a tensor that can then
# be manipulated as well
g = sj.gradients(cost, mu)
print(g)

# (Tensor: shape=(), dtype=float32)

# create the compiled function that will compute the cost and apply
# the update onto the variable
f = sj.function(outputs=cost, updates={mu:mu-0.2*g})

for i in range(10):
    print(f())

# 0.008471076
# 0.008201109
# 0.007946267
# ...
```

## Installation

Make sure to install all the needed GPU drivers (for GPU support, not mandatory) and install JAX as described in this [**guide**](https://symjax.readthedocs.io/en/latest/user/installation.html).