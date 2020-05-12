import symjax
import symjax.tensor as T
import matplotlib.pyplot as plt
import numpy as np

J = 8
Q = 1
scales = T.power(2,T.linspace(0, J, J*Q))
scales = scales[:, None]



wavelet = symjax.tensor.signal.complex_morlet(scales, 1/scales)
waveletw = symjax.tensor.signal.fourier_complex_morlet(scales, 1/scales, wavelet.shape[-1])



wavelet = wavelet.get()
waveletw = waveletw.get()
wavelet = wavelet / wavelet.max(1, keepdims=True)
plt.subplot(121)
for i in range(J*Q):
    plt.plot(i + np.abs(waveletw[i]), 'k')
    ff = np.abs(np.fft.fft(wavelet[i]))
    plt.plot(i + ff / ff.max(), '--r')

plt.subplot(122)
for i in range(J*Q):
    ff = np.fft.fftshift(np.real(np.fft.ifft(waveletw[i])))
    plt.plot(i + wavelet[i].real / wavelet[i].real.max(), c='k')
    plt.plot(i + ff / ff.max(), '--g')



plt.show()
