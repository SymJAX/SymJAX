#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
"""
Single-station covariance matrix
================================


This example shows how to calculate the interchannel spectral covariance matrix. It makes use of the obspy example trace available when installing obspy. This basic example does not apply any synchronization or pre-processing.

Considering a single seismic station with three channels (NS, EW, and Z), the covariance is of dimensions 3 times 3. The following example use a Fourier estimation window of 1 second and is estimated over 5 consecutive windows.
"""


import symjax
import symjax.tensor as T
import matplotlib.pyplot as plt
import numpy as np

J = 5
Q = 4
scales = T.power(2, T.linspace(0.1, J - 1, J * Q))
scales = scales[:, None]

print(symjax.tensor.get(scales))

wavelet = symjax.tensor.signal.complex_morlet(5 * scales, np.pi / scales)
waveletw = symjax.tensor.signal.fourier_complex_morlet(
    5 * scales, np.pi / scales, wavelet.shape[-1])
waveletlp = symjax.tensor.signal.littewood_paley_normalization(
    waveletw, down=np.pi / scales[-1, 0])

wavelet = symjax.tensor.get(wavelet)
waveletw = symjax.tensor.get(waveletw)
waveletlp = symjax.tensor.get(waveletlp)


plt.subplot(321)
for i in range(J * Q):
    fr = np.real(np.fft.fft(np.fft.ifftshift(wavelet[i])))
    fi = np.imag(np.fft.fft(np.fft.ifftshift(wavelet[i])))
    plt.plot(i + fr, '--b')
    plt.plot(i + fi, '--r')

plt.subplot(322)
for i in range(J * Q):
    plt.plot(2 * i + wavelet[i].real, c='b')
    plt.plot(2 * i + wavelet[i].imag, c='r')

plt.subplot(324)
for i in range(J * Q):
    fr = np.real(np.fft.fftshift(np.fft.ifft(waveletw[i])))
    fi = np.imag(np.fft.fftshift(np.fft.ifft(waveletw[i])))
    plt.plot(2 * i + fr / fr.max(), '--b')
    plt.plot(2 * i + fi / fi.max(), '--r')

plt.subplot(323)
for i in range(J * Q):
    plt.plot(i + waveletw[i].real, c='b')
    plt.plot(i + waveletw[i].imag, c='r')

plt.subplot(325)
for i in range(J * Q):
    plt.plot(i + waveletlp[i].real, c='b')
    plt.plot(i + waveletlp[i].imag, c='r')
plt.plot(np.abs(waveletlp).sum(0), c='g')

plt.subplot(326)
for i in range(J * Q):
    fr = np.real(np.fft.fftshift(np.fft.ifft(waveletlp[i])))
    fi = np.imag(np.fft.fftshift(np.fft.ifft(waveletlp[i])))
    plt.plot(2 * i + fr / fr.max(), '--b')
    plt.plot(2 * i + fi / fi.max(), '--r')
