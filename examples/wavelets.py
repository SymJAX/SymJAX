import symjax
import symjax.tensor as T
import matplotlib.pyplot as plt
import numpy as np

J = 5
Q = 4
scales = T.power(2,T.linspace(0.1, J - 1, J * Q))
scales = scales[:, None]

print(scales.get())

wavelet = symjax.tensor.signal.complex_morlet(5 * scales, np.pi /scales)
waveletw = symjax.tensor.signal.fourier_complex_morlet(5 * scales, np.pi /scales, wavelet.shape[-1])
waveletlp = symjax.tensor.signal.littewood_paley_normalization(waveletw, down = np.pi / scales[-1, 0])

wavelet = wavelet.get()
waveletw = waveletw.get()
waveletlp = waveletlp.get()



plt.subplot(321)
for i in range(J*Q):
    fr = np.real(np.fft.fft(wavelet[i]))
    fi = np.imag(np.fft.fft(wavelet[i]))
    plt.plot(i + fr, '--b')
    plt.plot(i + fi, '--r')

plt.subplot(322)
for i in range(J*Q):
    plt.plot(2*i + wavelet[i].real, c='b')
    plt.plot(2*i + wavelet[i].imag, c='r')

plt.subplot(324)
for i in range(J*Q):
    fr = np.real(np.fft.ifft(waveletw[i]))
    fi = np.imag(np.fft.ifft(waveletw[i]))
    plt.plot(2 * i + fr / fr.max(), '--b')
    plt.plot(2 * i + fi / fi.max(), '--r')

plt.subplot(323)
for i in range(J*Q):
    plt.plot(i + waveletw[i].real, c='b')
    plt.plot(i + waveletw[i].imag, c='r')

plt.subplot(325)
for i in range(J*Q):
    plt.plot(i + waveletlp[i].real, c='b')
    plt.plot(i + waveletlp[i].imag, c='r')
plt.plot(np.abs(waveletlp).sum(0), c='g')

plt.subplot(326)
for i in range(J*Q):
    fr = np.real(np.fft.ifft(waveletlp[i]))
    fi = np.imag(np.fft.ifft(waveletlp[i]))
    plt.plot(2 * i + fr / fr.max(), '--b')
    plt.plot(2 * i + fi / fi.max(), '--r')




plt.show()
