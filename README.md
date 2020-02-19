<div align="center">
<img src="https://raw.githubusercontent.com/RandallBalestriero/SymJAX/master/doc/img/logo.png" alt="logo"></img>
</div>

# SymJAX: symbolic CPU/GPU/TPU programming [![Test status](https://travis-ci.org/google/jax.svg?branch=master)](https://travis-ci.org/google/jax)

This is an under development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs.
[**Reference docs**](https://symjax.readthedocs.io/en/latest/)


## What is SymJAX ?

SymJAX is a symbolic programming version of JAX simplifying graph input/output/updates and providing additional functionalities for general machine learning and deep learning applications. From an user perspective SymJAX apparents to Theano with fast graph optimization/compilation and broad hardware support, along with Lasagne-like deep learning functionalities

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

# create the compield function that will compute the cost and apply
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

Make sure to install all the needed GPU drivers (for GPU support, not mandatory) and install JAX as follows (see [**guide**](https://github.com/google/jax/blob/master/README.md#installation)):

    # install jaxlib
    PYTHON_VERSION=cp37  # alternatives: cp35, cp36, cp37, cp38
    CUDA_VERSION=cuda92  # alternatives: cuda92, cuda100, cuda101, cuda102
    PLATFORM=linux_x86_64  # alternatives: linux_x86_64
    BASE_URL='https://storage.googleapis.com/jax-releases'
    pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.39-$PYTHON_VERSION-none-$PLATFORM.whl

    pip install --upgrade jax  # install jax

Then simply install SymJAX as follows:

    pip install symjax

once this is done, to leverage the dataset please set up the environment variable
    
    export DATASET_PATH=/path/to/default/location/
    
this path will be used as the default path where to download the various datasets in case no explicit path is given.
Additionally, the following options are standard to be set up to link with the CUDA library and deactivate the memory preallocation (example below for CUDA10.1, change for desired version)

    export CUDA_DIR="/usr/local/cuda-10.1"
    export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH 
    export LIBRARY_PATH=$CUDA_DIR/lib64:$LIBRARY_PATH                               
    export XLA_PYTHON_CLIENT_PREALLOCATE='false'                                
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_DIR"             
    
