import symjax
from symjax import nn
import symjax.tensor as T
import numpy as np

input = T.Placeholder((64, 3, 32, 32), "float32")
label = T.Placeholder((64,), "int32")
deterministic = T.Placeholder((), "bool")

block = nn.models.ResidualBlockv1(activation=nn.relu, dropout_rate=0.1)
transformed = nn.models.ResNet(
    input,
    widths=[32, 32, 64, 64, 128, 128],
    strides=[1, 1, 2, 1, 2, 1],
    block=block,
    deterministic=deterministic,
)

classifier = nn.layers.Dense(transformed, 10)
loss = nn.losses.sparse_softmax_crossentropy_logits(label, classifier).mean()

nn.optimizers.Adam(loss, 0.001)

train = symjax.function(
    input, label, deterministic, outputs=loss, updates=symjax.get_updates()
)
