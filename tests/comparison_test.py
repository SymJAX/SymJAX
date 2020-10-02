"""
Adam TF and SymJAX
==================

In this example we demonstrate how to perform a simple optimization with Adam in TF and SymJAX

"""

import symjax
import symjax.tensor as T
from symjax.nn import optimizers
import numpy as np
from tqdm import tqdm


np.random.seed(0)
BS = 1000
D = 500
X = np.random.randn(BS, D).astype("float32")
Y = X.dot(np.random.randn(D, 1).astype("float32")) + 2


def TF1(x, y, N, lr, model, preallocate=False):
    import tensorflow.compat.v1 as tf

    tf.compat.v1.disable_v2_behavior()
    tf.reset_default_graph()

    tf_input = tf.placeholder(dtype=tf.float32, shape=[BS, D])
    tf_output = tf.placeholder(dtype=tf.float32, shape=[BS, 1])

    np.random.seed(0)

    tf_W = tf.Variable(np.random.randn(D, 1).astype("float32"))
    tf_b = tf.Variable(
        np.random.randn(
            1,
        ).astype("float32")
    )

    tf_loss = tf.reduce_mean((tf.matmul(tf_input, tf_W) + tf_b - tf_output) ** 2)
    if model == "SGD":
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(tf_loss)
    elif model == "Adam":
        train_op = tf.train.AdamOptimizer(lr).minimize(tf_loss)

    # initialize session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    losses = []
    for i in tqdm(range(N)):
        losses.append(
            sess.run([tf_loss, train_op], feed_dict={tf_input: x, tf_output: y})[0]
        )

    return losses


def TF_EMA(X):
    import tensorflow.compat.v1 as tf

    tf.compat.v1.disable_v2_behavior()
    tf.reset_default_graph()
    x = tf.placeholder("float32")
    # Create an ExponentialMovingAverage object
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    op = ema.apply([x])
    out = ema.average(x)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer(), feed_dict={x: X[0]})

    outputs = []
    for i in range(len(X)):
        sess.run(op, feed_dict={x: X[i]})
        outputs.append(sess.run(out))
    return outputs


def SJ_EMA(X, debias=True):
    symjax.current_graph().reset()
    x = T.Placeholder((), "float32", name="x")
    value = symjax.nn.schedules.ExponentialMovingAverage(x, 0.9, debias=debias)[0]
    train = symjax.function(x, outputs=value, updates=symjax.get_updates())
    outputs = []
    for i in range(len(X)):
        outputs.append(train(X[i]))
    return outputs


def SJ(x, y, N, lr, model, preallocate=False):
    symjax.current_graph().reset()
    sj_input = T.Placeholder(dtype=np.float32, shape=[BS, D])
    sj_output = T.Placeholder(dtype=np.float32, shape=[BS, 1])

    np.random.seed(0)

    sj_W = T.Variable(np.random.randn(D, 1).astype("float32"))
    sj_b = T.Variable(
        np.random.randn(
            1,
        ).astype("float32")
    )

    sj_loss = ((sj_input.dot(sj_W) + sj_b - sj_output) ** 2).mean()

    if model == "SGD":
        optimizers.SGD(sj_loss, lr)
    elif model == "Adam":
        optimizers.Adam(sj_loss, lr)
    train = symjax.function(
        sj_input, sj_output, outputs=sj_loss, updates=symjax.get_updates()
    )

    losses = []
    for i in tqdm(range(N)):
        losses.append(train(x, y))

    return losses


def test_learn_bn():

    symjax.current_graph().reset()
    import tensorflow as tf
    from tensorflow.keras import layers
    import symjax.nn as nn

    np.random.seed(0)

    batch_size = 128

    W = np.random.randn(5, 5, 3, 2)
    W2 = np.random.randn(2 * 28 * 28, 10)

    inputs = layers.Input(shape=(3, 32, 32))
    out = layers.Permute((2, 3, 1))(inputs)
    out = layers.Conv2D(
        2, 5, activation="linear", kernel_initializer=lambda *args, **kwargs: W
    )(out)
    out = layers.BatchNormalization(-1)(out)
    out = layers.Activation("relu")(out)
    out = layers.Flatten()(out)
    out = layers.Dense(
        10, activation="linear", kernel_initializer=lambda *args, **kwargs: W2
    )(out)
    model = tf.keras.Model(inputs, out)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

    input = T.Placeholder((batch_size, 3, 32, 32), "float32")
    label = T.Placeholder((batch_size,), "int32")
    deterministic = T.Placeholder((), "bool")


#    conv = nn.layers.Conv2D(input, 2, (5, 5), W=W.transpose((3, 2, 0, 1)))

#    conv = nn.layers.Conv2D(input, 2, (5, 5))
#    out = nn.layers.BatchNormalization(conv, [1], deterministic=deterministic)
#    out = conv
#    out = nn.relu(out)
#    out = nn.layers.Dense(out.transpose((0, 2, 3, 1)), 10)#, W=W2)
    #out = nn.layers.Dense(input, 10)
    #loss = nn.losses.sparse_softmax_crossentropy_logits(label, out).mean()
    V2 = T.Variable(W2)
    loss = input.sum() * V2.sum()
    out=loss
    nn.optimizers.SGD(loss, 0.001, params=[V2])
    print(V2, loss, input, label, deterministic, out)
    f = symjax.function(input, label, deterministic, outputs=[loss, out])
    g = symjax.function(
        input,
        label,
        deterministic,
        outputs=symjax.gradients(loss, symjax.get_variables(trainable=True)),
    )
    train = symjax.function(
        input,
        label,
        deterministic,
        outputs=loss,
        updates=symjax.get_updates(),
    )

    for epoch in range(10):
        # generate some random inputs and labels
        x = np.random.randn(batch_size, 3, 32, 32)
        y = np.random.randint(0, 10, size=batch_size)

        # test predictions during testing mode
        preds = model(x, training=False)
        nb = np.isclose(preds, f(x, y, 1)[1], atol=1e-3).mean()
        print("preds not training", nb)
        assert nb > 0.8

        # test prediction during training mode
        # now get the gradients
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(y, preds)
            )

        nb = np.isclose(preds, f(x, y, 0)[1], atol=1e-3).mean()
        print("preds training", nb)
        assert nb > 0.8

        # test loss function during training
        losss = f(x, y, 0)[0]
        print("losses", losss, loss)
        assert np.abs(losss - loss) < 1e-3

        # test the gradients
        grads = tape.gradient(loss, model.trainable_variables)
        sj_grads = g(x, y, 0)

        for tf_g, sj_g in zip(grads, sj_grads):
            if sj_g.ndim == 4:
                sj_g = sj_g.transpose((2, 3, 1, 0))

            print(sj_g.shape, tf_g.shape)
            nb = np.isclose(
                np.reshape(sj_g, -1),
                np.reshape(tf_g, -1),
                atol=1e-3,
            ).mean()
            print("grads training", nb)
            assert nb >= 0.5

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train(x, y, 0)


def test_ema():
    np.random.seed(0)
    sample = np.random.randn(100)
    assert np.isclose(TF_EMA(sample), SJ_EMA(sample, True), atol=1e-3).mean() >= 0.45
    assert np.allclose(TF_EMA(sample), SJ_EMA(sample, False), atol=1e-7)


def test_adam():

    Ns = [400]
    lrs = [0.001, 0.01, 0.1]
    colors = ["r", "b", "g"]
    for k, N in enumerate(Ns):
        for c, lr in enumerate(lrs):
            loss = TF1(X, Y, N, lr, "Adam")
            assert np.allclose(
                TF1(X, Y, N, lr, "Adam"), SJ(X, Y, N, lr, "Adam"), atol=1e-03
            )
            assert np.allclose(
                TF1(X, Y, N, lr, "SGD"), SJ(X, Y, N, lr, "SGD"), atol=1e-03
            )


if __name__ == "__main__":
    test_learn_bn()
    test_ema()
    test_adam()
