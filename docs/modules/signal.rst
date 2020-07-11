 .. _symjax-tensor-signal:

:mod:`symjax.tensor.signal`
---------------------------

.. automodule:: symjax.tensor.signal


Apodization Windows
===================

.. autosummary::

   symjax.tensor.signal.blackman
   symjax.tensor.signal.bartlett
   symjax.tensor.signal.hamming
   symjax.tensor.signal.hanning
   symjax.tensor.signal.kaiser
   symjax.tensor.signal.tukey


Additional Time-Frequency Representations
=========================================

.. autosummary::
  symjax.tensor.signal.mfcc
  symjax.tensor.signal.dct
  symjax.tensor.signal.wvd
  symjax.tensor.signal.hilbert_transform

Filters (Banks)
===============

.. autosummary::
  symjax.tensor.signal.fourier_complex_morlet
  symjax.tensor.signal.complex_morlet
  symjax.tensor.signal.morlet
  symjax.tensor.signal.sinc_bandpass
  symjax.tensor.signal.mel_filterbank

Operations
==========

.. autosummary::
  symjax.tensor.signal.convolve
  symjax.tensor.signal.convolve2d
  symjax.tensor.signal.correlate
  symjax.tensor.signal.correlate2d


Detailed Descritpions
=====================

.. autofunction:: blackman
.. autofunction:: bartlett
.. autofunction:: hamming
.. autofunction:: hanning
.. autofunction:: kaiser


.. autofunction:: mfcc
.. autofunction:: dct
.. autofunction:: wvd
.. autofunction:: hilbert_transform
