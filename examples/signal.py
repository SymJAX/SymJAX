import time
import jax
import numpy as np
import sys
sys.path.insert(0, "../")
from scipy.io.wavfile import read

import theanoxla
import theanoxla.tensor as T

import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client


""" calculate the wigner ville distribution of an audio file """
def wvd(x, t=None, N=None, trace=0, make_analytic=True):
        
    if x.ndim == 1: [xrow, xcol] = np.shape(np.array([x]))
    else: raise ValueError("Signal x must be one-dimensional.")
        
    if t is None: t = np.arange(len(x))
    if N is None: N = len(x)
    
    if (N <= 0 ): raise ValueError("Number of Frequency bins N must be greater than zero.")
    
    if t.ndim == 1: [trow, tcol] = np.shape(np.array([t]))
    else: raise ValueError("Time indices t must be one-dimensional.")
    
    
    tfr = np.zeros([N, tcol], dtype='complex')
    for icol in range(0, tcol):
        ti = t[icol]
        taumax = min([ti, xcol-ti-1, int(round(N/2.0))-1])
        tau = np.arange(-taumax, taumax+1)
        indices = ((N+tau)%N)
        tfr[np.ix_(indices, [icol])] = np.transpose(np.array(x[ti+tau] * np.conj(x[ti-tau]), ndmin=2))
#        tau=int(round(N/2))+1
#        if ((ti+1) <= (xcol-tau)) and ((ti+1) >= (tau+1)):
#            if(tau >= tfr.shape[0]): tfr = append(tfr, zeros([1, tcol]), axis=0)
#            tfr[np.ix_([tau], [icol])] = np.array(0.5 * (x[ti+tau] * np.conj(x[ti-tau]) + x[ti-tau] * np.conj(x[ti+tau])))
    
    tfr = np.real(np.fft.fft(tfr, axis=0))
    return (tfr, t, f )

fs, SIGNAL = read('../a2002011001-e02.wav')
SIGNAL = SIGNAL[:, 0]
SIGNAL = SIGNAL / SIGNAL.max()

#plt.imshow(wvd(SIGNAL, N=4096)[0], aspect='auto')
#plt.show(block=True)
#adasdf



SS = 2**13
signal = T.Placeholder((SS,), 'float32')

mini_signal = T.Placeholder((16,), 'int32')
pp = T.extract_signal_patches(mini_signal, 5)
qq = T.fliplr(T.extract_signal_patches(mini_signal, 5))

get_patch = theanoxla.function(mini_signal, outputs=[T.concatenate([pp, qq], 1)])
print(get_patch(np.arange(16, dtype='int32'))[0])


def wvd(signal, h):
    p = T.extract_signal_patches(signal, h)
    pr = p * T.fliplr(p)
    qr = T.real(T.fft(T.cast(p,'complex64'), xla_client.FftType.FFT, (h,)))
    return qr

def spec(signal, h):
    p = T.extract_signal_patches(signal, h)
    qr = T.fft(T.cast(p,'complex64'), xla_client.FftType.FFT, (h,))
    return qr




output1 = wvd(signal, 4096)

X, Y = T.meshgrid(T.linspace(-5, 5, 8), T.linspace(-5, 5, 8))
Z = T.stack([X.flatten(), Y.flatten()], 1)
COV = T.Variable(np.random.rand(2,2), name='cov')
gaussian = T.exp(-(Z.dot(T.abs(COV))*Z).sum(1)).reshape((8, 8))
output1c = T.convNd(T.expand_dims(T.expand_dims(output1,0), 0),
                    T.expand_dims(T.expand_dims(gaussian, 0), 0) / gaussian.sum())

output2 = spec(signal, 4096)

f = theanoxla.function(signal, outputs = [output1])
g = theanoxla.function(signal, outputs = [output2])
h = theanoxla.function(signal, outputs = [output1c[0, 0]])


alls = list()
WW = 5
f(SIGNAL[: SS].astype('float32'))[0][:, 2048:]
g(SIGNAL[: SS].astype('float32'))[0][:, 2048:]

t = time.time()
for W in range(WW):
    alls.append(g(SIGNAL[W*SS:(W+1)*SS].astype('float32'))[0][:, 2048:])
print(time.time() - t)

alls = list()
t = time.time()
for W in range(WW):
    alls.append(f(SIGNAL[W*SS:(W+1)*SS].astype('float32'))[0][:, 2048:])
print(time.time() - t)


allsc = list()
t = time.time()
for W in range(WW):
    allsc.append(h(SIGNAL[W*SS:(W+1)*SS].astype('float32'))[0][:, 2048:])
print(time.time() - t)


plt.figure(figsize=(18, 5))
#print(u.shape)
plt.subplot(411)
plt.plot(SIGNAL[:WW*SS])
plt.subplot(412)
plt.imshow(np.log(np.abs(np.concatenate(alls, 0).T) + 0.000001), aspect='auto')
plt.subplot(413)
plt.imshow(np.log(np.abs(np.concatenate(allsc, 0).T) + 0.000001), aspect='auto')
plt.subplot(414)
t = time.time()
plt.specgram(SIGNAL[:WW*SS], noverlap=4095, NFFT=4096, Fs = 1)
print(time.time()-t)


plt.savefig('baseline_wvd.pdf')

