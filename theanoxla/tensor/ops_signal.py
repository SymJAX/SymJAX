import jax.numpy as jnp
import jax.lax as jla
import numpy

from .. import tensor as T
from jax.lib import xla_client

def mel_to_freq(m):
    # convert mel to frequency with
    # f = 700(10^{m/2595}-1)
    return 700 * (T.pow(10.,(m/2595))-1)

def freq_to_mel(f):
    # convert frequency to mel with
    # m = 2595(log_{10}(1+f/700)
    return 2595 * T.log10(1+f / 700)

def mel_filterbank(length, n_filter, low, high, nyquist):

    # convert the low and high frequency into mel scale
    low_freq_mel = freq_to_mel(low)
    high_freq_mel = freq_to_mel(high)

    # generate center frequencies uniformly spaced in Mel scale.
    mel_points = T.linspace(low_freq_mel, high_freq_mel, n_filter + 2)
    #print(mel_points.get({}))
    # turn them into frequency and thus becoming in log scale
    freq_points = mel_to_freq(mel_points)
    #print(freq_points.get({}))
    peaks = T.expand_dims(freq_points, 1)
    freqs = T.linspace(0, nyquist, length)
    print(freqs.get({}))
    filter_bank = T.hat_1D(freqs, peaks[:-2],
                           peaks[1:-1], peaks[2:])
    return filter_bank

def hamming(n):
    return 0.54 - 0.46 * T.cos(2*3.14159*T.linspace(0, 1, n))

def hanning(n):
    return 0.5 - 0.5 * T.cos(2*3.14159*T.linspace(0, 1, n))

def stft(signal, window, hop, apod=T.ones, nfft=None, mode='valid'):
    if nfft is None:
        nfft = window
    if mode == 'same':
        left = (window + 1) // 2
        psignal = T.pad(signal, [[0, 0], [0, 0], [left, window + 1 - left]])
    elif mode == 'full':
        left = (window + 1) //2
        psignal = T.pad(signal, [[0, 0], [0, 0], [window - 1, window - 1]])
    else:
        psignal = signal
    p = T.extract_signal_patches(psignal, window, hop) * apod(window).reshape((1, 1, -1))
    if nfft > window:
        pp = T.pad(p, [[0, 0], [0, 0], [0, 0], [0, nfft - window]])
    else:
        pp = p
    S = T.fft(T.cast(pp,'complex64'), xla_client.FftType.FFT, (nfft,))
    return S[..., nfft // 2:].transpose([0, 1, 3, 2])


def spectrogram(signal, window, hop, apod=T.ones, nfft=None, mode='valid'):
    return T.abs(stft(signal, window, hop, apod, nfft, mode)) ** 2.


def mfsc(signal, window, hop, n_filter, low_freq, high_freq, nyquist,
         nfft=None, mode='valid', apod=hanning):
    spec = spectrogram(signal, window, hop, apod, nfft, mode)
    filterbank = mel_filterbank(spec.shape[-2], n_filter, low_freq, high_freq,
                                nyquist)
    output = (T.expand_dims(spec, -3) * T.expand_dims(filterbank[::-1, ::-1], -1)).sum(-2)
    return output



def wvd(signal, h, hop, L, apod=T.ones, mode='valid'):
    # define the following constant for clarity
    PI = 2*3.14159

    # compute the stft
    s = stft(signal, h, hop, apod, nfft = 2 * h, mode=mode) #(N C F T)

    # remodulate the stft prior the spectral correlation for simplicity
    # with the following mask
    freq = T.linspace(0, 0.5, h).reshape((-1, 1))
    time = T.range(s.shape[-1])
    mask = T.complex(T.cos(PI*time*freq), T.sin(PI*time*freq))

    # extract vertical (freq) partches to perform auto correlation
    patches = T.extract_image_patches(s * mask, (2 * L + 1, 1), (2, 1),
                                      mode='same')[..., 0] #(N C F' T L)
    output = (patches * T.conj(T.flip(patches, -1))\
                     * hanning(2 * L + 1)).sum(-1)
    return T.real(output)


def fft(x, nfft=None):
    output = T.fft(T.cast(x,'complex64'), xla_client.FftType.FFT)
    return output





