.. jaxonn documentation master file, created by
   sphinx-quickstart on Wed Jan 29 09:29:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to SymJAX's documentation!
==================================

`XLA`_ + `Autograd`_ = `JAX`_

Symbolic programming with `JAX`_ = `SymJAX`_

`XLA`_: compiler that optimizes a computational graph fusing multiple kernels
into one preventing intermediate computation, reducing memory operations
and increasing performances.

`JAX`_: python interface providing a numpy-like software on top of XLA and
providing just-in-time compilation a well as advanced automatic
differenciation.

`SymJAX`_: symbolic programming version of `JAX`_ simplifying graph
input/output/updates and providing additional functionalities for general
machine learning and deep learning applications.


This is a research project, not an official product, expect bugs and sharp
edges; please help by trying it out, reporting bugs.


Examples
========


.. literalinclude:: ../examples/sgd.py


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


General API
===========

.. toctree::
  :maxdepth: 1

  modules/symjax
  modules/tensor
  modules/pdfs
  modules/signal
  modules/random

Deep Learning API
=================

.. toctree::
  :maxdepth: 1

  modules/datasets
  modules/initializers
  modules/layers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _XLA: https://www.tensorflow.org/xla
.. _Autograd: https://github.com/hips/autograd
.. _JAX: https://github.com/google/jax
.. _SymJAX: https://github.com/RandallBalestriero/SymJAX
