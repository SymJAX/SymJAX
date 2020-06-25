import sys
from .base import jax_wrap
import jax.numpy as jnp

# Add the fft functions

module = sys.modules[__name__]
names = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "fftfreq",
    "rfftfreq",
    "ifftshift",
    "fftshift",
]

for name in names:
    module.__dict__.update({name: jax_wrap(jnp.fft.__dict__[name])})
