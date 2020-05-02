Installation (GPU)
==================

Preriquisites
-------------

First install cuda/cudnn/GPU drivers.

JAX installation
----------------

please see `Jax Installation <https://github.com/google/jax#installation>`_ for
addition information, here is a minimal instruction to install the GPU version

PYTHON_VERSION=cp37  # alternatives: cp35, cp36, cp37, cp38
CUDA_VERSION=cuda92  # alternatives: cuda92, cuda100, cuda101, cuda102
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.38-$PYTHON_VERSION-none-$PLATFORM.whl


SymJAX installation
-------------------

pip install symjax


