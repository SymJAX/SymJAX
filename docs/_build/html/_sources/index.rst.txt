.. symjax documentation master file, created by
   sphinx-quickstart on Wed Jan 29 09:29:10 2020.


Welcome to SymJAX's documentation!
==================================

- `JAX`_ = `XLA`_ + `Autograd`_
- `SymJAX`_ =  `JAX`_ + symbolic programming + deep Learning

`XLA`_ is a compiler that optimizes a computational graph by fusing multiple kernels into one preventing intermediate computation, reducing memory operations and increasing performances.

`JAX`_ is a python interface that provides a `Numpy`_-like software on top of XLA and providing just-in-time compilation a well as advanced automatic differenciation.

`SymJAX`_ is a symbolic programming version of `JAX`_ simplifying graph input, output and updates and providing additional functionalities for general machine learning and deep learning applications. From an user perspective `SymJAX`_ apparents to `Theano`_ with fast graph optimization/compilation and broad hardware support, along with `Lasagne`_-like deep learning functionalities


This is an under development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs.


Contents
========

.. toctree::
  :maxdepth: 2

  user/installation
  user/wrap
  user/examples

API
===

General
-------

.. toctree::
  :maxdepth: 1

  modules/symjax
  modules/tensor
  modules/pdfs
  modules/signal
  modules/random
  modules/utils

Deep Learning
-------------

.. toctree::
  :maxdepth: 1

  modules/datasets
  modules/initializers
  modules/layers
  modules/optimizers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _XLA: https://www.tensorflow.org/xla
.. _Autograd: https://github.com/hips/autograd
.. _JAX: https://github.com/google/jax
.. _SymJAX: https://github.com/RandallBalestriero/SymJAX
.. _Theano: https://github.com/Theano/Theano
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _Numpy: https://numpy.org
