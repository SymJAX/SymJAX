import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def W(x, sigma):
    cov = [[sigma**2/2, 0], [0, 1/sigma ** 2]]
    return multivariate_normal.pdf(x, mean=[0, 0], cov=cov)

def W2(x, sigma, sigma2):
    cov = [[sigma**2/2, 0], [0, 1/sigma ** 2]]
    cov2 = [[1/(sigma2**2+0.000001), 0], [0, 10000]]
    return multivariate_normal.pdf(x, mean=[0, 0], cov=cov) *\
             multivariate_normal.pdf(x, mean=[0, 0], cov=cov2)





N = 200
x, y = np.meshgrid(np.linspace(-8, 8, N), np.linspace(-8, 8, N))
X = np.stack([x.flatten(), y.flatten()], 1)

SIGMAS = [0.5, 1.5, 3, 6]
for t, sigma in enumerate(SIGMAS):
    plt.subplot(1, len(SIGMAS), t + 1)
    plt.title(r'$\sigma='+str(sigma)+'$', fontsize=15)
    plt.imshow(W(X, sigma).reshape((N, N)), aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'Time $(t)$', fontsize=18)
    plt.ylabel(r'Freq. $(\omega)$', fontsize=18)



plt.figure()

SIGMAS = [0.5, 1.5, 3, 6]
SIGMAS2 = [0.0, 1, 2, 4]

cpt = 1
for sigma in SIGMAS:
    for sigma2 in SIGMAS2:
        plt.subplot(len(SIGMAS), len(SIGMAS), cpt)
        if sigma == 0.5:
            plt.title(r'$\sigma_{\omega}='+str(sigma2)+'$', fontsize=15)
        plt.imshow(W2(X, sigma, sigma2).reshape((N, N)), aspect='auto')
        plt.xticks([])
        plt.yticks([])
        cpt += 1
        if sigma2 == 0:
            plt.ylabel(r'$\sigma_{t}='+str(sigma)+'$', fontsize=15)

#        plt.xlabel(r'Time $(t)$', fontsize=18)
#        plt.ylabel(r'Freq. $(\omega)$', fontsize=18)

plt.show(block=True)
