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


def bin_to_freq(bins, max_f):
    return (bins / bins.max()) * max_f

def freq_to_bin(freq, n_bins, fmin, fmax):
    unit = (fmax-fmin) / n_bins
    return (freq / unit).astype('int32')


def mel_to_freq(m, option='linear'):
    # convert mel to frequency with
    if option == 'linear':
        # Fill in the linear scale
        f_sp = 200.0 / 3

        # And now the nonlinear scale
        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = min_log_hz / f_sp   # same (Mels)
        logstep = numpy.log(6.4) / 27.0                # step size for log region

        # If we have vector data, vectorize
        freq = min_log_hz * T.exp(logstep * (m - min_log_mel))
        return T.where( m >= min_log_mel, freq, f_sp * m)
    else:
        return 700 * (T.power(10.,(m/2595))-1)


def freq_to_mel(f, option='linear'):
    # convert frequency to mel with
    if option == 'linear':

        # linear part slope
        f_sp = 200.0 / 3

        # Fill in the log-scale part
        min_log_hz = 1000.0    # beginning of log region (Hz)
        min_log_mel = min_log_hz / f_sp   # same (Mels)
        logstep = numpy.log(6.4) / 27.0    # step size for log region
        mel = min_log_mel + T.log(f / min_log_hz) / logstep
        return T.where(f >= min_log_hz, mel, f/f_sp)
    else:
        return 2595 * T.log10(1+f / 700)


def mel_filterbank(length, n_filter, low, high, nyquist):

    # convert the low and high frequency into mel scale
    low_freq_mel = freq_to_mel(low)
    high_freq_mel = freq_to_mel(high)

    # generate center frequencies uniformly spaced in Mel scale.
    mel_points = T.linspace(low_freq_mel, high_freq_mel, n_filter + 2)
    # turn them into frequency and thus becoming in log scale
    freq_points = freq_to_bin(mel_to_freq(mel_points), length, 0, nyquist)
    peaks = T.expand_dims(freq_points, 1)
    freqs = T.range(length)
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
    return S[..., : int(numpy.ceil(nfft / 2))].transpose([0, 1, 3, 2])


def spectrogram(signal, window, hop, apod=hanning, nfft=None, mode='valid'):
    return T.abs(stft(signal, window, hop, apod, nfft, mode))


def melspectrogram(signal, window, hop, n_filter, low_freq, high_freq, nyquist,
         nfft=None, mode='valid', apod=hanning):
    spec = spectrogram(signal, window, hop, apod, nfft, mode)
    filterbank = mel_filterbank(spec.shape[-2], n_filter, low_freq, high_freq,
                                nyquist)
    flip_filterbank = filterbank.expand_dims(-1)
    output = (T.expand_dims(spec, -3) * flip_filterbank).sum(-2)
    return output

def mfcc(signal, window, hop, n_filter, low_freq, high_freq, nyquist, n_mfcc,
         nfft=None, mode='valid', apod=hanning):
    """
    https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#mfcc
    """
    tf = melspectrogram(signal, window, hop, n_filter, low_freq, high_freq,
                        nyquist, nfft, mode, apod)
    tf_db = power_to_db(tf)
    M = dct(tf_db, axes=(2,))
    return M


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """
    https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#power_to_db
    Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude
    """
    ref_value = numpy.abs(ref)
    log_spec = 10.0 * T.log10(T.maximum(amin, S)/T.maximum(amin, ref))
    if top_db is not None:
        if top_db < 0:
            error
        return T.maximum(log_spec, log_spec.max() - top_db)
    else:
        return log_spec


def wvd(signal, window, hop, L, apod=hanning, mode='valid'):
    # define the following constant for clarity
    PI = 2*3.14159

    # compute the stft with 2 times bigger window to interp.
    s = stft(signal, window, hop, apod, nfft = 2*window, mode=mode) #(N C F T)

    # remodulate the stft prior the spectral correlation for simplicity
    # with the following mask
    step = 1 / window
    freq = T.linspace(-step * L, step * L, 2 * L + 1)
    time = T.range(s.shape[-1]).reshape((-1, 1))
    mask = T.complex(T.cos(PI*time*freq), T.sin(PI*time*freq)) * hanning(2*L+1)

    # extract vertical (freq) partches to perform auto correlation
    patches = T.extract_image_patches(s, (2 * L + 1, 1), (2, 1),
                                      mode='same')[..., 0] #(N C F' T L)
    output = (patches * T.conj(T.flip(patches, -1)) * mask).sum(-1)
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


def dct(signal, axes=(-1,)):
    """
    https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
    """
    if len(axes) > 1:
        raise NotImplemented('not yet implemented more than 1D')
    to_pad = [(0,0) if ax not in axes else (0, signal.shape[ax])
                                   for ax in range(signal.ndim)]
    pad_signal = T.pad(signal, to_pad)
    exp = 2 * T.exp(-1j * 3.14159 * T.linspace(0, 0.5, signal.shape[axes[0]]))
    y = fft(pad_signal, axes=axes)
    cropped_y = T.dynamic_slice_in_dim(y, 0, signal.shape[axes[0]], axes[0])
    return T.real(cropped_y * exp.expand_dims(-1))


