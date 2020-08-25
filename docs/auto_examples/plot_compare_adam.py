"""
Adam TF and SymJAX
==================

In this example we demonstrate how to perform a simple optimization with Adam in TF and SymJAX

"""

import matplotlib.pyplot as plt

import symjax
import symjax.tensor as T
from symjax.nn import optimizers
import numpy as np
from tqdm import tqdm


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
    tf_b = tf.Variable(np.random.randn(1,).astype("float32"))

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
    print(x, value)
    print(symjax.current_graph().roots(value))
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
    sj_b = T.Variable(np.random.randn(1,).astype("float32"))

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


sample = np.random.randn(100)

plt.figure()
plt.plot(sample, label="Original signal", alpha=0.5)
plt.plot(TF_EMA(sample), c="orange", label="TF ema", linewidth=2, alpha=0.5)
plt.plot(SJ_EMA(sample), c="green", label="SJ ema (biased)", linewidth=2, alpha=0.5)
plt.plot(
    SJ_EMA(sample, False),
    c="green",
    linestyle="--",
    label="SJ ema (unbiased)",
    linewidth=2,
    alpha=0.5,
)
plt.legend()


plt.figure()
Ns = [400, 1000]
lrs = [0.001, 0.01, 0.1]
colors = ["r", "b", "g"]
for k, N in enumerate(Ns):
    plt.subplot(1, len(Ns), 1 + k)
    for c, lr in enumerate(lrs):
        loss = TF1(X, Y, N, lr, "Adam")
        plt.plot(loss, c=colors[c], linestyle="-", alpha=0.5)
        loss = SJ(X, Y, N, lr, "Adam")
        plt.plot(loss, c=colors[c], linestyle="--", alpha=0.5, linewidth=2)
        plt.title("lr:" + str(lr))
plt.suptitle("Adam Optimization quadratic loss (-:TF, --:SJ)")


plt.figure()
Ns = [400, 1000]
lrs = [0.001, 0.01, 0.1]
colors = ["r", "b", "g"]
for k, N in enumerate(Ns):
    plt.subplot(1, len(Ns), 1 + k)
    for c, lr in enumerate(lrs):
        loss = TF1(X, Y, N, lr, "SGD")
        plt.plot(loss, c=colors[c], linestyle="-", alpha=0.5)
        loss = SJ(X, Y, N, lr, "SGD")
        plt.plot(loss, c=colors[c], linestyle="--", alpha=0.5, linewidth=2)
        plt.title("lr:" + str(lr))
        plt.xlabel("steps")
plt.suptitle("GD Optimization quadratic loss (-:TF, --:SJ)")
plt.show()
