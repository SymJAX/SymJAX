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
# https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from scipy.ndimage import gaussian_filter


fs, SIGNAL = read("output2.wav")
SIGNAL = SIGNAL[2 ** 15 :, 0]
SIGNAL = SIGNAL / SIGNAL.max()


SS = 2 ** 16
signal = T.Placeholder((SS,), "float32")


signal2 = T.reshape(signal, (1, 1, -1))
wv = T.signal.wvd(signal2, 1024, 32, L=32, apod=T.signal.hanning, mode="same")
sp = T.signal.spectrogram(signal2, 256, 32, apod=T.signal.hanning, mode="same")
melsp = T.signal.melspectrogram(signal2, 1024, 32, 80, 10, 20000, 22000)
mfcc = T.signal.mfcc(signal2, 1024, 32, 80, 10, 20000, 22000, 12)

filters = T.signal.mel_filterbank(1024, 80, 10, 20000, 22000)
fil = theanoxla.function(outputs=[filters])
tfs = theanoxla.function(signal, outputs=[wv[0, 0], sp[0, 0], melsp[0, 0], mfcc[0, 0]])

t = time.time()
TFs = tfs(SIGNAL[:SS].astype("float32"))
FIL = fil()[0]
for i in range(80):
    plt.plot(FIL[i])
plt.show(block=True)
print(time.time() - t)


plt.figure(figsize=(18, 5))

ax0 = plt.subplot(511)
plt.plot(np.linspace(0, 1, len(SIGNAL[:SS])), SIGNAL[:SS])
plt.xticks([])

for i, name in enumerate(["wv", "sp", "melsp", "mfcc"]):
    plt.subplot(5, 1, 2 + i, sharex=ax0)
    plt.imshow((np.log(np.abs(TFs[i]) + 1e-8)), aspect="auto", extent=(0, 1, 0, 1))
    plt.xticks([])
    plt.title(name)


plt.tight_layout()
plt.show(block=True)
plt.savefig("baseline_wvd.pdf")
