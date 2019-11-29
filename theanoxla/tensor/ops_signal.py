import jax.numpy as jnp
import jax.lax as jla
import numpy
from .. import tensor as T
from .base import add_fn, Op


def hanning(M):
    """
    Return the Hanning window.
    The Hanning window is a taper formed by using a weighted cosine.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).
    See Also
    --------
    bartlett, blackman, hamming, kaiser
    Notes
    -----
    The Hanning window is defined as
    .. math::  w(n) = 0.5 - 0.5cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1
    The Hanning was named for Julius von Hann, an Austrian meteorologist.
    It is also known as the Cosine Bell. Some authors prefer that it be
    called a Hann window, to help avoid confusion with the very similar
    Hamming window.
    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.
    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.
    """
    n = T.arange(0, M)
    return 0.5 - 0.5*T.cos(2.0*3.14159*n/(M-1))


def hamming(M):
    """
    Return the Hamming window.
    The Hamming window is a taper formed by using a weighted cosine.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).
    See Also
    --------
    bartlett, blackman, hanning, kaiser
    Notes
    -----
    The Hamming window is defined as
    .. math::  w(n) = 0.54 - 0.46cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1
    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
    and is described in Blackman and Tukey. It was recommended for
    smoothing the truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.
    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.
    """
    n = T.arange(0, M)
    return 0.54 - 0.46*T.cos(2.0*3.14159*n/(M-1))

def mel_to_freq(m):
    # convert mel to frequency with
    # f = 700(10^{m/2595}-1)
    return 700 * (T.power(10.,(m/2595))-1)


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
    S = fft(pp, (nfft,), (-1,))
    return S[..., nfft // 2:].transpose([0, 1, 3, 2])


def spectrogram(signal, window, hop, apod=hanning, nfft=None, mode='valid'):
    return T.abs(stft(signal, window, hop, apod, nfft, mode)) ** 2.


def mfsc(signal, window, hop, n_filter, low_freq, high_freq, nyquist,
         nfft=None, mode='valid', apod=hanning):
    spec = spectrogram(signal, window, hop, apod, nfft, mode)
    filterbank = mel_filterbank(spec.shape[-2], n_filter, low_freq, high_freq,
                                nyquist)
    flip_filterbank = T.expand_dims(filterbank[::-1, ::-1], -1)
    output = (T.expand_dims(spec, -3) * flip_filterbank).sum(-2)
    return output


def wvd(signal, h, hop, L, apod=hanning, mode='valid'):
    # define the following constant for clarity
    PI = 2*3.14159

    # compute the stft with 2 times bigger window to interp.
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


class fft(Op):
    @staticmethod
    def fn(input, s=None, axes=(-1,)):
        if s is not None:
            assert len(s) == len(axes)
            # make then all positive (no -1 -2 -3 etc)
            axes = [a if a >= 0 else input.ndim + a for a in axes]
            # check how much to pad
            cpt = 0
            to_pad = []
            for d in range(input.ndim):
                if d in axes:
                    to_pad.append((0, s[cpt]-input.shape[d]))
                    cpt += 1
                else:
                    to_pad.append((0, 0))
            pad_input = jnp.pad(input, to_pad)
        else:
            pad_input = input
        return jnp.fft.fftn(pad_input, s=None, axes=axes)

class ifft(Op):
    pass
add_fn(ifft)(jnp.fft.ifftn)
