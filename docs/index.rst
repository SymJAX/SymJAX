.. symjax documentation master file, created by
   sphinx-quickstart on Wed Jan 29 09:29:10 2020.


Welcome to SymJAX's documentation!
==================================

.. centered:: **symbolic programming combining only the finest features of JAX, Theano, Lasagne and Tensorflow**


SymJAX = JAX+NetworkX

.. container:: twocol

    .. container:: leftside

        - `XLA`_ is a compiler that optimizes a computational graph by fusing multiple kernels into one preventing intermediate computation, reducing memory operations and increasing performances.

        - `JAX`_ is a `XLA`_ python interface that provides a `Numpy`_-like user experience with just-in-time compilation and `Autograd`_ powered automatic differenciation.

    .. container:: rightside

        - `NetworkX`_ is a Python package for the creation, manipulation, and study of directed and undirected graphs is a Python package for the creation, manipulation, and study of directed and undirected graphs is a Python package for the creation, manipulation, and study of directed and undirected graphs



`SymJAX`_ is a symbolic programming version of `JAX`_ 
providing a `Theano`_-like user experience thanks to a `NetworkX`_ powered computational graph backend. In addition of simplifying graph input/output, variable updates and graph utilities, `SymJAX`_ also features machine learning and deep learning tools similar to `Lasagne`_ and `Tensorflow1`_ .


*This is an under development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs and missing pieces.*


**Installation Guide** : :ref:`installation`

**Developer Guide** : :ref:`developer`


Modules
-------

We briefly describe below the structure of `SymJAX`_ and what are (in term of functionalities) the closest analog from other known libraries:

- :ref:`symjax-data` : everything related to downloading/importing/batchifying/patchifying datasets. Large corpus of time-series and computer vision dataset, similar to :py:mod:`tensorflow_datasets` with additional utilities

- :ref:`symjax-tensor` : everything related to operating with tensors (array like objects) similar to :py:mod:`numpy` and :py:mod:`theano.tensor`, specialized submodules are

  + :ref:`symjax-tensor-fft`: like :py:mod:`numpy.fft`
  + :ref:`symjax-tensor-signal`: like :py:mod:`scipy.signal` + additional time-frequency and wavelet tools
  + :ref:`symjax-tensor-random`: like :py:mod:`numpy.random`

- :ref:`symjax-nn` : everything related to machine/deep-learning mixing :py:mod:`lasagne`, :py:mod:`tensorflow`, :py:mod:`torch.nn` and :py:mod:`keras` and subdivided into

  + :ref:`symjax-nn-layers`: like :py:mod:`lasagne.layers` or :py:mod:`tf.keras.layers`
  + :ref:`symjax-nn-optimizers`: like :py:mod:`lasagne.optimizers` or :py:mod:`tf.keras.optimizers`
  + :ref:`symjax-nn-losses`: like :py:mod:`lasagne.losses` or :py:mod:`tf.keras.losses`
  + :ref:`symjax-nn-initializers`: like :py:mod:`lasagne.initializers` or :py:mod:`tf.keras.initializers`
  + :ref:`symjax-nn-schedules`: external variable state control (s.a. for learning rate schedules) as in :py:mod:`lasagne.initializers` or :py:mod:`tf.keras.optimizers.schedules` or `optax`_

- :ref:`symjax-probabilities` : like :py:mod:`tensorflow-probabilities`

- :ref:`symjax-rl` : like `tfagents`_ or OpenAI `SpinningUp`_ and `Baselines`_ (no environment is implemented as `Gym`_ already provides a large collection), submodules are

  + :py:mod:`symjax.rl.utils` providing utilities to interesact with environments, play, learn, buffers, ...
  + :py:mod:`symjax.rl.agents` providing the basic agents such as DDPG, PPO, DQN, ...

**Roadmap of incoming updates** : :ref:`roadmap`



Tutorials
---------

- :ref:`function`
- :ref:`clone`
- :ref:`none`
- :ref:`while`
- :ref:`saving`
- :ref:`viz`
- :ref:`wrapf`
- :ref:`wrapc`

Gallery
-------

- :ref:`gallery_basic`
- :ref:`gallery_nns`
- :ref:`gallery_datasets`
- :ref:`gallery_sp`


.. toctree::
  :hidden:
  :maxdepth: 1

  user/installation
  user/tutorial
  auto_examples/index
  user/developers


.. toctree::
  :hidden:
  :maxdepth: 1

  modules/symjax
  modules/data
  modules/tensor
  modules/interpolation
  modules/signal
  modules/fft
  modules/random
  modules/linalg
  modules/nn
  modules/initializers
  modules/layers
  modules/optimizers
  modules/schedules
  modules/losses
  modules/probabilities
  modules/rl



.. _XLA: https://www.tensorflow.org/xla
.. _Autograd: https://github.com/hips/autograd
.. _JAX: https://github.com/google/jax
.. _SymJAX: https://github.com/RandallBalestriero/SymJAX
.. _Theano: https://github.com/Theano/Theano
.. _Lasagne: https://github.com/Lasagne/Lasagne
.. _Numpy: https://numpy.org
.. _Tensorflow1: https://www.tensorflow.org/
.. _NetworkX: https://networkx.github.io/
.. _optax: https://github.com/deepmind/optax
.. _tfagents: https://www.tensorflow.org/agents/overview
.. _SpinningUp: https://openai.com/blog/spinning-up-in-deep-rl/
.. _Baselines: https://github.com/openai/baselines
.. _Gym: https://gym.openai.com/
