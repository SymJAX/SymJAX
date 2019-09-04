import jax
import jax.numpy as np
import numpy as NP

from .. import NewOp

cos = NewOp(np.cos)
add = NewOp(np.add)
sum = NewOp(np.sum)
sub = NewOp(np.subtract)
mul = NewOp(np.multiply)

