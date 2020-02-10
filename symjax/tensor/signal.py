import jax.numpy as jnp
import jax.numpy.fft as jnpf
import jax.lax as jla
import numpy
import inspect
from .. import tensor as T
from .base import Op, jax_wrap
import sys


################ Add the apodization windows

names = ['blackman',
         'bartlett',
         'hamming',
         'hanning',
         'kaiser']

module = sys.modules[__name__]
for name in names:
    module.__dict__.update({name: jax_wrap(jnp.__dict__[name])})

################# Add the fft functions into signal

names = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn', 'rfft', 'irfft',
         'rfft2', 'irfft2', 'rfftn', 'irfftn', 'fftfreq', 'rfftfreq']
for name in names:
    print(name)
    module.__dict__.update({name: jax_wrap(jnpf.__dict__[name])})



################# Add some utility functions


def bin_to_freq(bins, max_f):
    return (bins / bins.max()) * max_f


def freq_to_bin(freq, n_bins, fmin, fmax):
    unit = (fmax - fmin) / n_bins
    return (freq / unit).astype('int32')


def mel_to_freq(m, option='linear'):
    # convert mel to frequency with
    if option == 'linear':
        # Fill in the linear scale
        f_sp = 200.0 / 3

        # And now the nonlinear scale
        # beginning of log region (Hz)
        min_log_hz = 1000.0
        min_log_mel = min_log_hz / f_sp   # same (Mels)
        # step size for log region
        logstep = numpy.log(6.4) / 27.0

        # If we have vector data, vectorize
        freq = min_log_hz * T.exp(logstep * (m - min_log_mel))
        return T.where(m >= min_log_mel, freq, f_sp * m)
    else:
        return 700 * (T.power(10., (m / 2595)) - 1)


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
        return T.where(f >= min_log_hz, mel, f / f_sp)
    else:
        return 2595 * T.log10(1 + f / 700)


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
    log_spec = 10.0 * T.log10(T.maximum(amin, S) / T.maximum(amin, ref))
    if top_db is not None:
        if top_db < 0:
            error
        return T.maximum(log_spec, log_spec.max() - top_db)
    else:
        return log_spec


#################### Now some filter-bank and additional Time-Frequency Repr.


def sinc_bandpass(time, f0, f1):
    high = f0 * T.sinc(time * f0)
    low = f1 * T.sinc(time * f1)
    return 2 * (high - low)




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
    """
    Compute the Shoft-Time-Fourier-Transform of a signal given the
    window length, hop and additional parameters.

    Parameters
    ----------

        signal: array
            the signal (possibly stacked of signals)

        window: int
            the window length to be considered for the fft

        hop: int
            the amount by which the window is moved

        apod: func
            a function that takes an integer as input and return
            the apodization window of the same length

        nfft: int (optional)
            the number of bin that the fft on the window will use.
            If not given it is set the same as window.

        mode: 'valid', 'same' or 'full'
            the padding of the input signals

    Returns
    -------

        output: complex array
            the complex stft
    """
    if nfft is None:
        nfft = window
    if mode == 'same':
        left = (window + 1) // 2
        psignal = T.pad(signal, [[0, 0], [0, 0], [left, window + 1 - left]])
    elif mode == 'full':
        left = (window + 1) // 2
        psignal = T.pad(signal, [[0, 0], [0, 0], [window - 1, window - 1]])
    else:
        psignal = signal

    apodization = apod(window).reshape((1, 1, -1))

    p = T.extract_signal_patches(
        psignal, window, hop) * apodization
    S = fft(pp, (nfft,))
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


def wvd(signal, window, hop, L, apod=hanning, mode='valid'):
    # define the following constant for clarity
    PI = 2 * 3.14159

    # compute the stft with 2 times bigger window to interp.
    s = stft(signal, window, hop, apod, nfft=2 * window, mode=mode)

    # remodulate the stft prior the spectral correlation for simplicity
    # with the following mask
    step = 1 / window
    freq = T.linspace(-step * L, step * L, 2 * L + 1)
    time = T.range(s.shape[-1]).reshape((-1, 1))
    mask = T.complex(T.cos(PI * time * freq),
                     T.sin(PI * time * freq)) * hanning(2 * L + 1)

    # extract vertical (freq) partches to perform auto correlation
    patches = T.extract_image_patches(s, (2 * L + 1, 1), (2, 1),
                                      mode='same').squeeze()  # (N C F' T L)
    output = (patches * T.conj(T.flip(patches, -1)) * mask).sum(-1)
    return T.real(output)


def dct(signal, axes=(-1,)):
    """
    https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft
    """
    if len(axes) > 1:
        raise NotImplemented('not yet implemented more than 1D')
    to_pad = [(0, 0) if ax not in axes else (0, signal.shape[ax])
              for ax in range(signal.ndim)]
    pad_signal = T.pad(signal, to_pad)
    exp = 2 * T.exp(-1j * 3.14159 * T.linspace(0, 0.5, signal.shape[axes[0]]))
    y = fft(pad_signal, axes=axes)
    cropped_y = T.dynamic_slice_in_dim(y, 0, signal.shape[axes[0]], axes[0])
    return T.real(cropped_y * exp.expand_dims(-1))
