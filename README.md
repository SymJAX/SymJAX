<div align="center">
<img src="https://raw.githubusercontent.com/RandallBalestriero/SymJAX/master/doc/img/logo.png" alt="logo"></img>
</div>

# SymJAX: symbolic CPU/GPU/TPU programming [![Test status](https://travis-ci.org/google/jax.svg?branch=master)](https://travis-ci.org/google/jax)

[**Reference docs**](https://symjax.readthedocs.io/en/latest/)


## What is SymJAX ?

SymJAX is a symbolic programming version of JAX simplifying graph input/output/updates and providing additional functionalities for general machine learning and deep learning applications. From an user perspective SymJAX apparents to Theano with fast graph optimization/compilation and broad hardware support, along with Lasagne-like deep learning functionalities

## Examples

```python
import sys
sys.path.insert(0, "../")
import symjax
import symjax.tensor as T

# create our variable to be optimized
mu = T.Variable(T.random.normal((), seed=1))

# create our cost
cost = T.exp(-(mu-1)**2)

# get the gradient, notice that it is itself a tensor that can then
# be manipulated as well
g = symjax.gradients(cost, mu)
print(g)

# (Tensor: shape=(), dtype=float32)

# create the compield function that will compute the cost and apply
# the update onto the variable
f = symjax.function(outputs=cost, updates={mu:mu-0.2*g})

for i in range(10):
    print(f())

# 0.008471076
# 0.008201109
# 0.007946267
# 0.007705368
# 0.0074773384
# 0.007261208
# 0.0070561105
# 0.006861261
# 0.006675923
# 0.006499458
```

## Installation

conda list --explicit > spec-file.txt
conda create --name myenv --file spec-file.txt

    pip freeze>requirements.txt


    pip download -r requirements.txt -d path_to_the_folder


    pip install -r requirements.txt --find-links=path_to_the_folder


export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH 
export LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LIBRARY_PATH                
export DATASET_PATH='XXX'                                    
export XLA_PYTHON_CLIENT_PREALLOCATE='false'                                export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-10.1"             export CUDA_DIR="/usr/local/cuda-10.1"
