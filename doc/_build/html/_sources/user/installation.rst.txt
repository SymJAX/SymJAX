Installation
============

This installation is restricted to GPU support only.


Installation with pip
---------------------

1. Install all GPU divers and compilers (``cuda``, ``cudnn``, and GPU drivers).

2. Install ``jax`` following `Jax Installation <https://github.com/google/jax#installation>`_. Here is a minimal instruction to install the GPU version

    .. code-block:: bash

        $ PYTHON_VERSION=cp37  # alternatives: cp35, cp36, cp37, cp38
        $ CUDA_VERSION=cuda92  # alternatives: cuda92, cuda100, cuda101, cuda102
        $ PLATFORM=linux_x86_64  # alternatives: linux_x86_64
        $ BASE_URL='https://storage.googleapis.com/jax-releases'
        $ pip install --upgrade $BASE_URL/$CUDA_VERSION jaxlib-0.1.38-$PYTHON_VERSION-none-$PLATFORM.whl


3. Install SymJAX with

    .. code-block:: bash

        $ pip install symjax

Manual installation
-------------------

1. Clone this repository with

    .. code-block:: bash

        $ git clone https://github.com/RandallBalestriero/SymJAX

2. Install.

    .. code-block:: bash

        $ cd SymJAX
        $ pip install -r requirements.txt
        $ pip install .




