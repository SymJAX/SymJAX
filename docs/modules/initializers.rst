 .. _symjax-nn-initializers:

:mod:`symjax.nn.initializers`
-----------------------------


This module provides all the basic initializers used in Deep Learning. All the involved operations are meant to take as input
a shape of the desired weight tensor (vector, matrix, ...) and will return a numpy-array.

.. automodule:: symjax.nn.initializers

.. autosummary::
  
  constant
  uniform
  normal
  orthogonal
  glorot_uniform
  glorot_normal
  he_uniform
  he_normal
  lecun_uniform
  get_fans
  variance_scaling

Detailed Descriptions
=====================

.. autofunction:: constant
.. autofunction:: uniform
.. autofunction:: normal
.. autofunction:: orthogonal
.. autofunction:: glorot_uniform
.. autofunction:: glorot_normal
.. autofunction:: he_uniform
.. autofunction:: he_normal
.. autofunction:: lecun_uniform
.. autofunction:: get_fans
.. autofunction:: variance_scaling