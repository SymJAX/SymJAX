.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_compare_adam.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_compare_adam.py:


Adam TF and SymJAX
==================

In this example we demonstrate how to perform a simple optimization with Adam in TF and SymJAX



.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_001.svg
          :alt: plot compare adam
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_002.svg
          :alt: Adam Optimization quadratic loss (-:TF, --:SJ), lr:0.1, lr:0.1, lr:0.1, lr:0.1, lr:0.1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_003.svg
          :alt: GD Optimization quadratic loss (-:TF, --:SJ), lr:0.1, lr:0.1, lr:0.1, lr:0.1, lr:0.1
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/vrael/anaconda3/lib/python3.7/site-packages/jax/lib/xla_bridge.py:125: UserWarning: No GPU/TPU found, falling back to CPU.
      warnings.warn('No GPU/TPU found, falling back to CPU.')
    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=true_divide, fn=true_divide, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Variable(name=num_steps, shape=(), dtype=int32, trainable=False, scope=/ExponentialMovingAverage/), Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/), Placeholder(name=x, shape=(), dtype=float32, scope=/)]
    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=add, fn=add, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Placeholder(name=x, shape=(), dtype=float32, scope=/), Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/)]
      0%|          | 0/10 [00:00<?, ?it/s]    100%|##########| 10/10 [00:00<00:00, 475.07it/s]
      0%|          | 0/10 [00:00<?, ?it/s]     10%|#         | 1/10 [00:00<00:01,  5.64it/s]    100%|##########| 10/10 [00:00<00:00, 54.83it/s]
      0%|          | 0/10 [00:00<?, ?it/s]    100%|##########| 10/10 [00:00<00:00, 615.52it/s]
      0%|          | 0/10 [00:00<?, ?it/s]     10%|#         | 1/10 [00:00<00:01,  6.19it/s]    100%|##########| 10/10 [00:00<00:00, 60.02it/s]
      0%|          | 0/10 [00:00<?, ?it/s]    100%|##########| 10/10 [00:00<00:00, 640.44it/s]
      0%|          | 0/10 [00:00<?, ?it/s]     10%|#         | 1/10 [00:00<00:01,  5.99it/s]    100%|##########| 10/10 [00:00<00:00, 58.11it/s]
      0%|          | 0/100 [00:00<?, ?it/s]    100%|##########| 100/100 [00:00<00:00, 1213.85it/s]
      0%|          | 0/100 [00:00<?, ?it/s]      1%|1         | 1/100 [00:00<00:18,  5.43it/s]    100%|##########| 100/100 [00:00<00:00, 425.01it/s]
      0%|          | 0/100 [00:00<?, ?it/s]    100%|##########| 100/100 [00:00<00:00, 1410.19it/s]
      0%|          | 0/100 [00:00<?, ?it/s]      1%|1         | 1/100 [00:00<00:16,  6.14it/s]    100%|##########| 100/100 [00:00<00:00, 475.00it/s]
      0%|          | 0/100 [00:00<?, ?it/s]    100%|##########| 100/100 [00:00<00:00, 1705.90it/s]
      0%|          | 0/100 [00:00<?, ?it/s]      1%|1         | 1/100 [00:00<00:16,  6.14it/s]    100%|##########| 100/100 [00:00<00:00, 475.06it/s]
      0%|          | 0/200 [00:00<?, ?it/s]    100%|##########| 200/200 [00:00<00:00, 2016.41it/s]
      0%|          | 0/200 [00:00<?, ?it/s]      0%|          | 1/200 [00:00<00:32,  6.13it/s]    100%|##########| 200/200 [00:00<00:00, 780.76it/s]
      0%|          | 0/200 [00:00<?, ?it/s]     98%|#########8| 197/200 [00:00<00:00, 1967.60it/s]    100%|##########| 200/200 [00:00<00:00, 1966.81it/s]
      0%|          | 0/200 [00:00<?, ?it/s]      0%|          | 1/200 [00:00<00:32,  6.09it/s]    100%|##########| 200/200 [00:00<00:00, 778.36it/s]
      0%|          | 0/200 [00:00<?, ?it/s]    100%|##########| 200/200 [00:00<00:00, 1993.66it/s]    100%|##########| 200/200 [00:00<00:00, 1988.50it/s]
      0%|          | 0/200 [00:00<?, ?it/s]      0%|          | 1/200 [00:00<00:32,  6.09it/s]    100%|##########| 200/200 [00:00<00:00, 770.55it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     51%|#####1    | 205/400 [00:00<00:00, 2045.31it/s]    100%|##########| 400/400 [00:00<00:00, 2212.99it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:04,  6.17it/s]     50%|####9     | 199/400 [00:00<00:22,  8.80it/s]    100%|##########| 400/400 [00:00<00:00, 1110.82it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     51%|#####1    | 205/400 [00:00<00:00, 2046.30it/s]    100%|##########| 400/400 [00:00<00:00, 2205.32it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:07,  5.91it/s]     52%|#####2    | 210/400 [00:00<00:22,  8.43it/s]    100%|##########| 400/400 [00:00<00:00, 1118.04it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     49%|####8     | 195/400 [00:00<00:00, 1945.89it/s]    100%|##########| 400/400 [00:00<00:00, 2130.75it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:10,  5.63it/s]     50%|####9     | 199/400 [00:00<00:25,  8.04it/s]    100%|##########| 400/400 [00:00<00:00, 1083.67it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#1        | 118/1000 [00:00<00:00, 1171.75it/s]     25%|##5       | 250/1000 [00:00<00:00, 1212.45it/s]     39%|###8      | 387/1000 [00:00<00:00, 1253.67it/s]     52%|#####2    | 521/1000 [00:00<00:00, 1278.24it/s]     66%|######5   | 658/1000 [00:00<00:00, 1303.24it/s]     79%|#######9  | 793/1000 [00:00<00:00, 1316.92it/s]     93%|#########3| 930/1000 [00:00<00:00, 1330.90it/s]    100%|##########| 1000/1000 [00:00<00:00, 1328.18it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:43,  6.10it/s]     21%|##1       | 210/1000 [00:00<01:30,  8.70it/s]     43%|####3     | 431/1000 [00:00<00:45, 12.41it/s]     65%|######5   | 654/1000 [00:00<00:19, 17.68it/s]     86%|########6 | 862/1000 [00:00<00:05, 25.17it/s]    100%|##########| 1000/1000 [00:00<00:00, 1592.35it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#2        | 124/1000 [00:00<00:00, 1237.42it/s]     26%|##5       | 258/1000 [00:00<00:00, 1266.35it/s]     39%|###9      | 394/1000 [00:00<00:00, 1290.27it/s]     53%|#####3    | 530/1000 [00:00<00:00, 1309.51it/s]     67%|######6   | 666/1000 [00:00<00:00, 1323.82it/s]     81%|########  | 807/1000 [00:00<00:00, 1345.78it/s]     94%|#########3| 939/1000 [00:00<00:00, 1337.00it/s]    100%|##########| 1000/1000 [00:00<00:00, 1334.79it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:46,  6.01it/s]     20%|##        | 203/1000 [00:00<01:32,  8.58it/s]     41%|####      | 409/1000 [00:00<00:48, 12.24it/s]     62%|######1   | 619/1000 [00:00<00:21, 17.44it/s]     84%|########3 | 835/1000 [00:00<00:06, 24.82it/s]    100%|##########| 1000/1000 [00:00<00:00, 1553.18it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     13%|#2        | 129/1000 [00:00<00:00, 1282.86it/s]     27%|##7       | 271/1000 [00:00<00:00, 1318.59it/s]     41%|####      | 408/1000 [00:00<00:00, 1333.34it/s]     55%|#####4    | 546/1000 [00:00<00:00, 1345.71it/s]     68%|######8   | 684/1000 [00:00<00:00, 1354.84it/s]     82%|########2 | 823/1000 [00:00<00:00, 1364.73it/s]     96%|#########6| 960/1000 [00:00<00:00, 1364.39it/s]    100%|##########| 1000/1000 [00:00<00:00, 1366.47it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:45,  6.05it/s]     21%|##1       | 210/1000 [00:00<01:31,  8.63it/s]     42%|####2     | 424/1000 [00:00<00:46, 12.31it/s]     64%|######4   | 643/1000 [00:00<00:20, 17.54it/s]     85%|########5 | 852/1000 [00:00<00:05, 24.96it/s]    100%|##########| 1000/1000 [00:00<00:00, 1575.02it/s]
      0%|          | 0/10 [00:00<?, ?it/s]    100%|##########| 10/10 [00:00<00:00, 688.37it/s]
      0%|          | 0/10 [00:00<?, ?it/s]     10%|#         | 1/10 [00:00<00:01,  7.92it/s]    100%|##########| 10/10 [00:00<00:00, 76.70it/s]
      0%|          | 0/10 [00:00<?, ?it/s]    100%|##########| 10/10 [00:00<00:00, 661.74it/s]
      0%|          | 0/10 [00:00<?, ?it/s]     10%|#         | 1/10 [00:00<00:01,  7.97it/s]    100%|##########| 10/10 [00:00<00:00, 77.16it/s]
      0%|          | 0/10 [00:00<?, ?it/s]    100%|##########| 10/10 [00:00<00:00, 646.42it/s]
      0%|          | 0/10 [00:00<?, ?it/s]     10%|#         | 1/10 [00:00<00:01,  8.08it/s]    100%|##########| 10/10 [00:00<00:00, 78.32it/s]
      0%|          | 0/100 [00:00<?, ?it/s]    100%|##########| 100/100 [00:00<00:00, 1304.67it/s]
      0%|          | 0/100 [00:00<?, ?it/s]      1%|1         | 1/100 [00:00<00:12,  7.92it/s]    100%|##########| 100/100 [00:00<00:00, 637.51it/s]
      0%|          | 0/100 [00:00<?, ?it/s]    100%|##########| 100/100 [00:00<00:00, 1555.66it/s]
      0%|          | 0/100 [00:00<?, ?it/s]      1%|1         | 1/100 [00:00<00:12,  7.85it/s]    100%|##########| 100/100 [00:00<00:00, 625.83it/s]
      0%|          | 0/100 [00:00<?, ?it/s]    100%|##########| 100/100 [00:00<00:00, 1343.49it/s]
      0%|          | 0/100 [00:00<?, ?it/s]      1%|1         | 1/100 [00:00<00:12,  7.85it/s]    100%|##########| 100/100 [00:00<00:00, 622.01it/s]
      0%|          | 0/200 [00:00<?, ?it/s]     72%|#######2  | 144/200 [00:00<00:00, 1432.24it/s]    100%|##########| 200/200 [00:00<00:00, 1479.19it/s]
      0%|          | 0/200 [00:00<?, ?it/s]      0%|          | 1/200 [00:00<00:25,  7.85it/s]    100%|##########| 200/200 [00:00<00:00, 1065.12it/s]
      0%|          | 0/200 [00:00<?, ?it/s]    100%|##########| 200/200 [00:00<00:00, 2092.05it/s]
      0%|          | 0/200 [00:00<?, ?it/s]      0%|          | 1/200 [00:00<00:24,  8.14it/s]    100%|##########| 200/200 [00:00<00:00, 1093.22it/s]
      0%|          | 0/200 [00:00<?, ?it/s]     74%|#######4  | 148/200 [00:00<00:00, 1477.98it/s]    100%|##########| 200/200 [00:00<00:00, 1488.81it/s]
      0%|          | 0/200 [00:00<?, ?it/s]      0%|          | 1/200 [00:00<00:25,  7.93it/s]    100%|##########| 200/200 [00:00<00:00, 1072.72it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     50%|#####     | 201/400 [00:00<00:00, 2004.73it/s]    100%|##########| 400/400 [00:00<00:00, 2101.32it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<00:51,  7.79it/s]     81%|########  | 323/400 [00:00<00:06, 11.12it/s]    100%|##########| 400/400 [00:00<00:00, 1577.96it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     35%|###5      | 140/400 [00:00<00:00, 1396.17it/s]     72%|#######1  | 286/400 [00:00<00:00, 1414.06it/s]    100%|##########| 400/400 [00:00<00:00, 1447.48it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:03,  6.31it/s]     70%|######9   | 279/400 [00:00<00:13,  9.01it/s]    100%|##########| 400/400 [00:00<00:00, 1318.46it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     42%|####2     | 170/400 [00:00<00:00, 1694.05it/s]     90%|######### | 360/400 [00:00<00:00, 1749.08it/s]    100%|##########| 400/400 [00:00<00:00, 1796.65it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:05,  6.08it/s]     56%|#####6    | 226/400 [00:00<00:20,  8.68it/s]    100%|##########| 400/400 [00:00<00:00, 1172.85it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     18%|#7        | 175/1000 [00:00<00:00, 1746.50it/s]     37%|###6      | 368/1000 [00:00<00:00, 1795.30it/s]     55%|#####5    | 550/1000 [00:00<00:00, 1800.54it/s]     73%|#######2  | 729/1000 [00:00<00:00, 1797.08it/s]     90%|########9 | 897/1000 [00:00<00:00, 1759.48it/s]    100%|##########| 1000/1000 [00:00<00:00, 1778.30it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:42,  6.15it/s]     24%|##3       | 239/1000 [00:00<01:26,  8.78it/s]     49%|####9     | 494/1000 [00:00<00:40, 12.52it/s]     76%|#######5  | 755/1000 [00:00<00:13, 17.85it/s]    100%|#########9| 999/1000 [00:00<00:00, 25.42it/s]    100%|##########| 1000/1000 [00:00<00:00, 1773.22it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#2        | 124/1000 [00:00<00:00, 1238.31it/s]     27%|##6       | 268/1000 [00:00<00:00, 1291.28it/s]     42%|####2     | 421/1000 [00:00<00:00, 1354.40it/s]     57%|#####6    | 568/1000 [00:00<00:00, 1386.44it/s]     72%|#######1  | 717/1000 [00:00<00:00, 1414.25it/s]     87%|########7 | 874/1000 [00:00<00:00, 1456.70it/s]    100%|##########| 1000/1000 [00:00<00:00, 1459.39it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:45,  6.05it/s]     27%|##6       | 268/1000 [00:00<01:24,  8.64it/s]     53%|#####3    | 533/1000 [00:00<00:37, 12.32it/s]     79%|#######9  | 792/1000 [00:00<00:11, 17.56it/s]    100%|##########| 1000/1000 [00:00<00:00, 1822.34it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#2        | 123/1000 [00:00<00:00, 1227.51it/s]     27%|##6       | 266/1000 [00:00<00:00, 1281.16it/s]     41%|####1     | 412/1000 [00:00<00:00, 1329.31it/s]     55%|#####5    | 551/1000 [00:00<00:00, 1346.77it/s]     68%|######8   | 684/1000 [00:00<00:00, 1340.22it/s]     81%|########1 | 814/1000 [00:00<00:00, 1326.07it/s]     94%|#########4| 944/1000 [00:00<00:00, 1316.61it/s]    100%|##########| 1000/1000 [00:00<00:00, 1343.70it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:43,  6.11it/s]     26%|##5       | 258/1000 [00:00<01:25,  8.71it/s]     52%|#####2    | 522/1000 [00:00<00:38, 12.43it/s]     78%|#######8  | 784/1000 [00:00<00:12, 17.72it/s]    100%|##########| 1000/1000 [00:00<00:00, 1811.44it/s]
    /home/vrael/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())






|


.. code-block:: default


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
    Ns = [10, 100, 200, 400, 1000]
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
    Ns = [10, 100, 200, 400, 1000]
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


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  17.988 seconds)


.. _sphx_glr_download_auto_examples_plot_compare_adam.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_compare_adam.py <plot_compare_adam.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_compare_adam.ipynb <plot_compare_adam.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
