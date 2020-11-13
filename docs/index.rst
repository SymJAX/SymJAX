.. symjax documentation master file, created by
   sphinx-quickstart on Wed Jan 29 09:29:10 2020.


Welcome to SymJAX's documentation!
==================================

.. centered:: **SymJAX = JAX+NetworkX**
  
.. raw:: html

    <div class="row" style="text-align:center">
      <div class="column" style="background-color:#bbb;">
        <h2 style="text-align:center">JAX</h2>
        <p><a href="https://github.com/google/jax">JAX</a> is a <a href="https://www.tensorflow.org/xla">XLA</a> python interface that provides a Numpy-like user experience with just-in-time compilation and <a href="https://github.com/hips/autograd">Autograd</a> powered automatic differenciation. <a href="https://www.tensorflow.org/xla">XLA</a> is a compiler that optimizes a computational graph by fusing multiple kernels into one preventing intermediate computation, reducing memory operations and increasing performances.</p>
      </div>
      <div class="column" style="background-color:#bbb;">
        <h2 style="text-align:center">NetworkX</h2>
        <p><a href="https://networkx.github.io/">NetworkX</a> is a Python package for the creation, manipulation, and study of directed and undirected graphs is a Python package for the creation, manipulation, and study of directed and undirected graphs is a Python package for the creation, manipulation, and study of directed and undirected graphs.</p>
      </div>
    </div>


.. raw:: html

  <p style="text-align:justify"><a href="https://github.com/RandallBalestriero/SymJAX">SymJAX</a> is a symbolic programming version of <a href="https://github.com/google/jax">JAX</a> 
  providing a <a href="https://github.com/Theano/Theano">Theano</a>-like user experience thanks to a <a href="https://networkx.github.io/">NetworkX</a> powered computational graph backend. In addition of simplifying graph input/output, variable updates and graph utilities, <a href="https://github.com/RandallBalestriero/SymJAX">SymJAX</a> also features machine learning and deep learning tools similar to <a href="https://github.com/Lasagne/Lasagne">Lasagne</a> and <a href="https://www.tensorflow.org/">Tensorflow1</a> as well as a lazy on-the-go execution capability like <a href="https://pytorch.org/">PyTorch</a> and <a href="https://www.tensorflow.org/tutorials/quickstart/beginner">Tensorflow2</a>.</p>

.. raw:: html

  <p style="text-align:justify;font-style:italic">This is an under development research project, not an official product, expect bugs and sharp edges; please help by trying it out, reporting bugs and missing pieces.</p>


.. centered:: **Installation Guide** : :ref:`installation`

.. centered:: **Implementation Walkthrough** : :ref:`walkthrough`

.. centered:: **Developer Guide** : :ref:`developer`

.. centered:: **Updates Roadmap** : :ref:`roadmap`


Modules
-------

We briefly describe below the structure of `SymJAX`_ and what are (in term of functionalities) the closest analog from other known libraries:

- :ref:`symjax-data` : everything related to downloading/importing/batchifying/patchifying datasets. Large corpus of time-series and computer vision dataset, similar to :py:mod:`tensorflow_datasets` with additional utilities

- :ref:`symjax-tensor` : everything related to operating with tensors (array like objects) similar to :py:mod:`numpy` and :py:mod:`theano.tensor`, specialized submodules are

  + :ref:`symjax-tensor-linalg`: like :py:mod:`scipy.linalg` and :py:mod:`numpy.linalg` 
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

  + :py:mod:`symjax.rl.utils` providing utilities to interact with environments, play, learn, buffers, ...
  + :py:mod:`symjax.rl.agents` providing the basic agents such as DDPG, PPO, DQN, ...



Tutorials
---------

SymJAX
''''''

- :ref:`function`
- :ref:`clone`
- :ref:`none`
- :ref:`while`
- :ref:`saving`
- :ref:`viz`
- :ref:`wrapf`
- :ref:`wrapc`
- :ref:`function`

Amortized Variational Inference
'''''''''''''''''''''''''''''''

- :ref:`basic_avi`

Reinforcement Learning
''''''''''''''''''''''

- :ref:`rl_notations`


Gallery
-------

- :ref:`gallery_basic`
- :ref:`gallery_nns`
- :ref:`gallery_datasets`
- :ref:`gallery_sp`


.. toctree::
  :hidden:
  :maxdepth: 2

  user/installation
  user/tutorials
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
.. _PyTorch: https://pytorch.org/
.. _Tensorflow2: https://www.tensorflow.org/tutorials/quickstart/beginner
