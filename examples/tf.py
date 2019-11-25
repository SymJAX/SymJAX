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
from scipy.ndimage import gaussian_filter


fs, SIGNAL = read('output2.wav')
SIGNAL = SIGNAL[2**15:, 0]
SIGNAL = SIGNAL / SIGNAL.max()


SS = 2**16
signal = T.Placeholder((SS,), 'float32')


signal2 = T.reshape(signal, (1, 1, -1))
wv = T.signal.wvd(signal2, 2048, 32, L=32, apod=T.signal.hanning, mode='same')
sp = T.signal.spectrogram(signal2, 256, 32, apod=T.signal.hanning, mode='same')
mfsc = T.signal.mfsc(signal2, 1024, 32, 80, 10, 20000, 22000)
fb = T.signal.mel_filterbank(512, 80, 2, 44100/2/2, 44100/2/2)
sp2 = T.signal.spectrogram(signal2, 2048, 32, apod=T.signal.hanning, mode='same')

wvf = theanoxla.function(signal, outputs = [wv[0, 0]])
spf = theanoxla.function(signal, outputs = [sp[0, 0]])
mfscf = theanoxla.function(signal, outputs = [mfsc[0, 0], fb])
spf2 = theanoxla.function(signal, outputs = [sp2[0, 0]])

#h = theanoxla.function(signal, outputs = [output1c[0, 0]])


#print(f(SIGNAL[: SS].astype('float32')))
#print(g(SIGNAL[: SS].astype('float32')))

WW = 1
t = time.time()
allwv = wvf(SIGNAL[:SS].astype('float32'))[0]
print(time.time() - t)

t = time.time()
allsp = spf(SIGNAL[:SS].astype('float32'))[0]
print(time.time() - t)

t = time.time()
allmfsc, fbfb = mfscf(SIGNAL[:SS].astype('float32'))
print(time.time() - t)

t = time.time()
allsp2 = spf2(SIGNAL[:SS].astype('float32'))[0]
print(time.time() - t)




plt.figure()
for i in range(80):
    plt.plot(fbfb[i])

plt.show(block=True)

plt.figure(figsize=(18, 5))
#print(u.shape)



ax0 = plt.subplot(511)
plt.plot(np.linspace(0, 1, len(SIGNAL[:WW*SS])),
         SIGNAL[:WW*SS])
plt.xticks([])
plt.subplot(512, sharex=ax0)
plt.imshow((np.log(allsp)), aspect='auto', extent=(0, 1, 0, 1))
plt.xticks([])                                                                                         
plt.title('SP 256')
plt.subplot(513, sharex=ax0)
plt.imshow((np.log(allsp2)), aspect='auto', extent=(0, 1, 0, 1))
plt.xticks([])                                                                                         
plt.title('SP 2048')
plt.subplot(514, sharex=ax0)
plt.imshow(np.log(abs(allwv)), aspect='auto', extent=(0, 1, 0, 1))
plt.xticks([])                                                                                         
plt.title('W 2048')
plt.subplot(515, sharex=ax0)
plt.imshow(np.log(abs(allmfsc)), aspect='auto', extent=(0, 1, 0, 1))
plt.xticks([])                                                                                         
plt.title('W 2048')





plt.tight_layout()
plt.show(block=True)
plt.savefig('baseline_wvd.pdf')

