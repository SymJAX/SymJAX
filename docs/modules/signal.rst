 .. _symjax-tensor-signal:

:mod:`symjax.tensor.signal`
---------------------------

.. automodule:: symjax.tensor.signal


Apodization Windows
===================

.. autosummary::

  blackman
  bartlett
  hamming
  hanning
  kaiser
  tukey


Additional Time-Frequency Representations
=========================================

.. autosummary::
  mfcc
  stft
  dct
  wvd
  hilbert_transform

Filters (Banks)
===============

.. autosummary::
  fourier_complex_morlet
  complex_morlet
  sinc_bandpass
  mel_filterbank

Operations
==========

.. autosummary::
  convolve
  batch_convolve
  convolve2d
  correlate
  correlate2d
  batch_pool


Detailed Descritpions
=====================

.. autofunction:: blackman
.. autofunction:: bartlett
.. autofunction:: hamming
.. autofunction:: hanning
.. autofunction:: kaiser
.. autofunction:: tukey

.. autofunction:: mfcc
.. autofunction:: stft
.. autofunction:: dct
.. autofunction:: wvd
.. autofunction:: hilbert_transform

.. autofunction:: fourier_complex_morlet
.. autofunction:: complex_morlet
.. autofunction:: sinc_bandpass
.. autofunction:: mel_filterbank

.. autofunction:: convolve
.. autofunction:: batch_convolve
.. autofunction:: convolve2d
.. autofunction:: correlate
.. autofunction:: correlate2d
.. autofunction:: batch_pool