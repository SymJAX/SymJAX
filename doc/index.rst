.. jaxonn documentation master file, created by
   sphinx-quickstart on Wed Jan 29 09:29:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to SymJAX's documentation!
==================================

XLA + Autograd = JAX
JAX + Symbolic = SymJAX



The advantages of a symbolic programming language are multiple. First, one
can create and test the entire computational graph without performing any
memory allocation or actual computations. Second, for performances, the
computational graph needs to be defined prior being optimized and compiled.
This holds across softwares (Tensorflow, JAX, ...).
SymJAX simply allows the user to do so in a simple, natural and
streamlined fashion.


Examples
========



Installation (GPU)
==================

Preriquisites
-------------

First install cuda/cudnn/GPU drivers.
Ensure that the GPUs that are visible.


JAX installation
----------------

PYTHON_VERSION=cp37  # alternatives: cp35, cp36, cp37, cp38
CUDA_VERSION=cuda92  # alternatives: cuda92, cuda100, cuda101, cuda102
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.38-$PYTHON_VERSION-none-$PLATFORM.whl


SymJAX installation
-------------------

pip install symjax


API
===

.. toctree::
  :maxdepth: 1

  modules/datasets
  modules/symjax
  modules/tensor
  modules/pdfs
  modules/signal


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
