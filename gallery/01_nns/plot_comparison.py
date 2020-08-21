"""
RNN/GRU example
===========

example of vanilla RNN for time series regression
"""
import symjax.tensor as T
from symjax import nn
import symjax
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(3500)


def classif_tf(train_x, train_y, test_x, test_y, mlp=True):
    import tensorflow as tf
    from tensorflow.keras import layers

    batch_size = 128

    inputs = layers.Input(shape=(3, 32, 32))
    if not mlp:
        out = layers.Permute((2, 3, 1))(inputs)
        out = layers.Conv2D(32, 3, activation="relu")(out)
        for i in range(3):
            for j in range(3):

                conv = layers.Conv2D(
                    32 * (i + 1), 3, activation="linear", padding="SAME"
                )(out)
                bn = layers.BatchNormalization(axis=-1)(conv)
                relu = layers.Activation("relu")(bn)
                conv = layers.Conv2D(
                    32 * (i + 1), 3, activation="linear", padding="SAME"
                )(relu)
                bn = layers.BatchNormalization(axis=-1)(conv)

                out = layers.Add()([out, bn])
            out = layers.AveragePooling2D()(out)
            out = layers.Conv2D(32 * (i + 2), 1, activation="linear")(out)

        out = layers.GlobalAveragePooling2D()(out)
    else:
        out = layers.Flatten()(inputs)
        for i in range(6):
            out = layers.Dense(4000, activation="linear")(out)
            bn = layers.BatchNormalization(axis=-1)(out)
            out = layers.Activation("relu")(bn)
    outputs = layers.Dense(10, activation="linear")(out)

    model = tf.keras.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    for epoch in range(5):
        accu = 0
        for x, y in symjax.data.utils.batchify(
            train_x, train_y, batch_size=batch_size, option="random"
        ):
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(y, preds)
                )
                print(preds)
                sdfg
            accu += tf.reduce_mean(tf.cast(y == tf.argmax(preds, 1), "float32"))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print("training", accu / (len(train_x) // batch_size))
        accu = 0
        for x, y in symjax.data.utils.batchify(
            test_x, test_y, batch_size=batch_size, option="continuous"
        ):
            preds = model(x, training=False)
            accu += tf.reduce_mean(tf.cast(y == tf.argmax(preds, 1), "float32"))
        print(accu / (len(test_x) // batch_size))


def classif_sj(train_x, train_y, test_x, test_y, mlp=True):
    symjax.current_graph().reset()
    from symjax import nn

    batch_size = 128

    input = T.Placeholder((batch_size, 3, 32, 32), "float32")
    labels = T.Placeholder((batch_size,), "int32")
    deterministic = T.Placeholder((), "bool")

    if not mlp:
        out = nn.relu(nn.layers.Conv2D(input, 32, (3, 3)))
        for i in range(3):
            for j in range(3):
                conv = nn.layers.Conv2D(out, 32 * (i + 1), (3, 3), pad="SAME")
                bn = nn.layers.BatchNormalization(
                    conv, [1], deterministic=deterministic
                )
                bn = nn.relu(bn)
                conv = nn.layers.Conv2D(bn, 32 * (i + 1), (3, 3), pad="SAME")
                bn = nn.layers.BatchNormalization(
                    conv, [1], deterministic=deterministic
                )
                out = out + bn

            out = nn.layers.Pool2D(out, (2, 2), pool_type="AVG")
            out = nn.layers.Conv2D(out, 32 * (i + 2), (1, 1))

        out = nn.layers.Pool2D(out, out.shape[-2:], pool_type="AVG")
    else:
        out = input
        for i in range(6):
            out = nn.layers.Dense(out, 4000)
            out = nn.relu(
                nn.layers.BatchNormalization(out, [1], deterministic=deterministic)
            )

    outputs = nn.layers.Dense(out, 10)

    loss = nn.losses.sparse_softmax_crossentropy_logits(labels, outputs).mean()
    nn.optimizers.Adam(loss, 0.001)

    accu = T.equal(outputs.argmax(1), labels).astype("float32").mean()

    train = symjax.function(
        input,
        labels,
        deterministic,
        outputs=[loss, accu, outputs],
        updates=symjax.get_updates(),
    )
    test = symjax.function(input, labels, deterministic, outputs=accu)

    for epoch in range(5):
        accu = 0
        for x, y in symjax.data.utils.batchify(
            train_x, train_y, batch_size=batch_size, option="random"
        ):
            accu += train(x, y, 0)[1]

        print("training", accu / (len(train_x) // batch_size))

        accu = 0
        for x, y in symjax.data.utils.batchify(
            test_x, test_y, batch_size=batch_size, option="continuous"
        ):
            accu += test(x, y, 1)
        print(accu / (len(test_x) // batch_size))


mnist = symjax.data.cifar10()
train_x, train_y = mnist["train_set/images"], mnist["train_set/labels"]
test_x, test_y = mnist["test_set/images"], mnist["test_set/labels"]
train_x /= train_x.max()
test_x /= test_x.max()


# classif_tf(train_x, train_y, test_x, test_y)
# training tf.Tensor(0.5272436, shape=(), dtype=float32)
# tf.Tensor(0.24088542, shape=(), dtype=float32)
# training tf.Tensor(0.71652645, shape=(), dtype=float32)
# tf.Tensor(0.6938101, shape=(), dtype=float32)
# training tf.Tensor(0.7882212, shape=(), dtype=float32)
# tf.Tensor(0.7083333, shape=(), dtype=float32)
# training tf.Tensor(0.82624197, shape=(), dtype=float32)
# tf.Tensor(0.7421875, shape=(), dtype=float32)
# training tf.Tensor(0.85697114, shape=(), dtype=float32)
# tf.Tensor(0.7845553, shape=(), dtype=float32)

classif_sj(train_x, train_y, test_x, test_y)
classif_tf(train_x, train_y, test_x, test_y)
# training 0.43359375
# 0.36157852564102566
# training 0.5697115384615384
# 0.41776842948717946
# training 0.6573116987179487
# 0.5833333333333334
# training 0.7119991987179487
# 0.6381209935897436
# training 0.7483173076923076
# 0.6378205128205128
