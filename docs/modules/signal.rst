 .. _symjax-tensor-signal:

:mod:`symjax.tensor.signal`
---------------------------

Implementation of various signal processing related techniques such as
time-frequency representations convolution/correlation/pooling operations,
as well as various apodization windows and filter-banks creations.



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


Time-Frequency Representations
==============================

.. autosummary::
  mfcc
  stft
  dct
  wvd
  hilbert_transform

Filters
=======

.. autosummary::
  fourier_complex_morlet
  complex_morlet
  sinc_bandpass
  mel_filterbank
  hat_1d

Convolution/Correlation/Pooling
===============================

.. autosummary::
  convolve
  batch_convolve
  convolve2d
  correlate
  correlate2d
  batch_pool


Utilities
===================

.. autosummary::
  extract_signal_patches
  extract_image_patches


Detailed Descriptions
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
.. autofunction:: hat_1d

.. autofunction:: convolve
.. autofunction:: batch_convolve
.. autofunction:: convolve2d
.. autofunction:: correlate
.. autofunction:: correlate2d
.. autofunction:: batch_pool


.. autofunction:: extract_signal_patches
.. autofunction:: extract_image_patches