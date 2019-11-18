import jax.numpy as jnp
import jax.lax as jla
import numpy

from .. import tensor as T
from jax.lib import xla_client

def mel_to_freq(m):
    # convert mel to frequency with
    # f = 700(10^{m/2595}-1)
    return 700 * (10 ** (m/2595)-1)

def freq_to_mel(f):
    # convert frequency to mel with
    # m = 2595(log_{10}(1+f/700)
    return 2595 * numpy.log10(1+f / 700)




def MFSC(NFFT=1024, N_FILT=40, low=0, high = 1):
    low_freq_mel = 0
    high_freq_mel = freq_to_mel(sampling_rate)
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = mel_to_freq(mel_points)
    peaks = numpy.floor((NFFT + 1) * hz_points / sample_rate)
    
    fbank = numpy.zeros((NFILT, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
    
        bounds = [int(peaks[m-1]), int(peaks[m])]
        bounds2 = [int(peaks[m]), int(peaks[m+1])]
        r = np.range(bounds[0], bounds[1])
        fbank[m - 1, bounds[0]:bounds[1]] = (r - bounds[0]) / (bounds[1] - bounds[0])
        r = r + bounds[1]
        fbank[m - 1, bounds2[0]:bounds2[1]] = (bounds2[1] - r) / (bounds2[1] - bounds2[0])
        filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

def hamming(n):
    return 0.54 - 0.46 * T.cos(2*3.14159*T.linspace(0, 1, n))

def hanning(n):
    return 0.5 - 0.5 * T.cos(2*3.14159*T.linspace(0, 1, n))


def stft(signal, window, hop, apod=T.ones, nfft=None, mode='valid'):
    if nfft is None:
        nfft = window
    if mode == 'same':
        left = window + 1 // 2
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
    return S[...,:nfft // 2].transpose([0, 1, 3, 2])


def spectrogram(signal, window, hop, apod=T.ones, nfft=None, mode='valid'):
    return T.abs(stft(signal, window, hop, apod, nfft, mode))


def wvd2(signal, h, hop, L, apod=T.ones, mode='valid'):
    pi = 2*3.14159
    s = stft(signal, h, hop, apod, nfft = 2 * h, mode=mode) #(N C F T)
#    freq = T.linspace(- L ,L , 2 * L + 1) / (1 * h)
#    time = T.range(s.shape[-1]).reshape((-1, 1))
    freq = T.linspace(0, 0.5, h).reshape((-1, 1))
    time = T.range(s.shape[-1])
    mask = T.complex(T.cos(pi*time*freq), T.sin(pi*time*freq))
    patches = T.extract_image_patches(s * mask, (2 * L + 1, 1), (2, 1),
                                      mode='same')[..., 0] #(N C F' T L)
#    mask = hanning(2 * L + 1)
    Ws = (patches * T.conj(T.flip(patches, -1)) * hanning(2 * L + 1)).sum(-1)
    return T.real(Ws)#PWD



def wvd(signal, h, hop, apod=T.ones):
    apod =  apod(h).reshape((1, 1, -1))
    p = T.extract_signal_patches(signal, h, hop)
    pr = p * T.flip(p, 3) * apod#[..., h:]
    qr = T.fft(T.cast(pr,'complex64'), xla_client.FftType.FFT, (h,))
    return T.real(T.transpose(qr[..., h // 2:], [0, 1, 3, 2]))





