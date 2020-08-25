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
          :alt: Adam Optimization quadratic loss (-:TF, --:SJ), lr:0.1, lr:0.1
          :class: sphx-glr-multi-img

    *

      .. image:: /auto_examples/images/sphx_glr_plot_compare_adam_003.svg
          :alt: GD Optimization quadratic loss (-:TF, --:SJ), lr:0.1, lr:0.1
          :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=true_divide, fn=true_divide, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Variable(name=num_steps, shape=(), dtype=int32, trainable=False, scope=/ExponentialMovingAverage/), Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/), Placeholder(name=x, shape=(), dtype=float32, scope=/)]
    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=add, fn=add, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Placeholder(name=x, shape=(), dtype=float32, scope=/), Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/)]
      0%|          | 0/400 [00:00<?, ?it/s]     29%|##9       | 117/400 [00:00<00:00, 1167.66it/s]     66%|######6   | 264/400 [00:00<00:00, 1243.99it/s]    100%|##########| 400/400 [00:00<00:00, 1359.29it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:58,  3.38it/s]     16%|#6        | 65/400 [00:00<01:09,  4.81it/s]     32%|###1      | 127/400 [00:00<00:39,  6.85it/s]     48%|####8     | 193/400 [00:00<00:21,  9.75it/s]     65%|######5   | 260/400 [00:00<00:10, 13.84it/s]     83%|########3 | 333/400 [00:00<00:03, 19.61it/s]    100%|##########| 400/400 [00:00<00:00, 448.34it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     31%|###1      | 125/400 [00:00<00:00, 1244.15it/s]     71%|#######   | 283/400 [00:00<00:00, 1327.70it/s]    100%|##########| 400/400 [00:00<00:00, 1454.29it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:29,  4.45it/s]     19%|#8        | 75/400 [00:00<00:51,  6.35it/s]     36%|###6      | 146/400 [00:00<00:28,  9.03it/s]     53%|#####3    | 213/400 [00:00<00:14, 12.83it/s]     70%|#######   | 282/400 [00:00<00:06, 18.18it/s]     86%|########6 | 346/400 [00:00<00:02, 25.65it/s]    100%|##########| 400/400 [00:00<00:00, 488.27it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     30%|###       | 120/400 [00:00<00:00, 1196.38it/s]     70%|######9   | 278/400 [00:00<00:00, 1288.65it/s]    100%|##########| 400/400 [00:00<00:00, 1430.54it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:33,  4.25it/s]     17%|#7        | 69/400 [00:00<00:54,  6.05it/s]     33%|###3      | 133/400 [00:00<00:31,  8.61it/s]     49%|####9     | 196/400 [00:00<00:16, 12.23it/s]     66%|######6   | 266/400 [00:00<00:07, 17.34it/s]     82%|########2 | 328/400 [00:00<00:02, 24.48it/s]     97%|#########7| 388/400 [00:00<00:00, 34.36it/s]    100%|##########| 400/400 [00:00<00:00, 465.16it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#1        | 119/1000 [00:00<00:00, 1189.00it/s]     28%|##7       | 276/1000 [00:00<00:00, 1281.73it/s]     43%|####3     | 430/1000 [00:00<00:00, 1347.66it/s]     58%|#####8    | 580/1000 [00:00<00:00, 1389.45it/s]     73%|#######3  | 734/1000 [00:00<00:00, 1429.67it/s]     89%|########8 | 887/1000 [00:00<00:00, 1456.83it/s]    100%|##########| 1000/1000 [00:00<00:00, 1481.06it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:46,  4.40it/s]      7%|7         | 71/1000 [00:00<02:28,  6.28it/s]     14%|#3        | 139/1000 [00:00<01:36,  8.93it/s]     21%|##        | 209/1000 [00:00<01:02, 12.69it/s]     28%|##7       | 276/1000 [00:00<00:40, 17.98it/s]     34%|###4      | 340/1000 [00:00<00:26, 25.37it/s]     40%|####      | 401/1000 [00:00<00:16, 35.60it/s]     46%|####6     | 464/1000 [00:00<00:10, 49.64it/s]     53%|#####2    | 526/1000 [00:01<00:06, 68.54it/s]     59%|#####8    | 586/1000 [00:01<00:04, 92.97it/s]     64%|######4   | 645/1000 [00:01<00:02, 123.53it/s]     70%|#######   | 702/1000 [00:01<00:01, 160.98it/s]     76%|#######6  | 765/1000 [00:01<00:01, 207.11it/s]     83%|########2 | 830/1000 [00:01<00:00, 260.12it/s]     89%|########9 | 894/1000 [00:01<00:00, 316.39it/s]     96%|#########6| 960/1000 [00:01<00:00, 374.16it/s]    100%|##########| 1000/1000 [00:01<00:00, 549.01it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#1        | 118/1000 [00:00<00:00, 1176.67it/s]     26%|##6       | 260/1000 [00:00<00:00, 1238.82it/s]     41%|####1     | 413/1000 [00:00<00:00, 1311.99it/s]     56%|#####6    | 565/1000 [00:00<00:00, 1366.31it/s]     70%|#######   | 705/1000 [00:00<00:00, 1375.46it/s]     86%|########5 | 856/1000 [00:00<00:00, 1413.09it/s]    100%|##########| 1000/1000 [00:00<00:00, 1436.30it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:44,  4.45it/s]      7%|7         | 70/1000 [00:00<02:26,  6.34it/s]     14%|#4        | 141/1000 [00:00<01:35,  9.02it/s]     21%|##1       | 212/1000 [00:00<01:01, 12.82it/s]     28%|##8       | 282/1000 [00:00<00:39, 18.17it/s]     35%|###5      | 352/1000 [00:00<00:25, 25.67it/s]     42%|####2     | 423/1000 [00:00<00:15, 36.10it/s]     49%|####9     | 494/1000 [00:00<00:10, 50.47it/s]     56%|#####6    | 563/1000 [00:01<00:06, 69.89it/s]     63%|######2   | 629/1000 [00:01<00:03, 94.16it/s]     70%|######9   | 696/1000 [00:01<00:02, 126.82it/s]     76%|#######6  | 763/1000 [00:01<00:01, 167.50it/s]     83%|########2 | 827/1000 [00:01<00:00, 214.51it/s]     89%|########9 | 891/1000 [00:01<00:00, 261.41it/s]     96%|#########5| 956/1000 [00:01<00:00, 318.45it/s]    100%|##########| 1000/1000 [00:01<00:00, 570.27it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#1        | 118/1000 [00:00<00:00, 1177.08it/s]     27%|##7       | 272/1000 [00:00<00:00, 1264.32it/s]     43%|####2     | 426/1000 [00:00<00:00, 1335.04it/s]     57%|#####7    | 575/1000 [00:00<00:00, 1377.03it/s]     72%|#######2  | 725/1000 [00:00<00:00, 1411.05it/s]     87%|########6 | 868/1000 [00:00<00:00, 1415.34it/s]    100%|##########| 1000/1000 [00:00<00:00, 1443.79it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:49,  4.35it/s]      6%|6         | 65/1000 [00:00<02:30,  6.20it/s]     13%|#3        | 133/1000 [00:00<01:38,  8.82it/s]     20%|##        | 203/1000 [00:00<01:03, 12.53it/s]     27%|##7       | 272/1000 [00:00<00:40, 17.76it/s]     34%|###4      | 342/1000 [00:00<00:26, 25.10it/s]     40%|####      | 403/1000 [00:00<00:16, 35.23it/s]     47%|####7     | 471/1000 [00:00<00:10, 49.23it/s]     54%|#####3    | 537/1000 [00:01<00:06, 68.14it/s]     60%|######    | 600/1000 [00:01<00:04, 92.88it/s]     66%|######6   | 662/1000 [00:01<00:02, 124.56it/s]     72%|#######2  | 724/1000 [00:01<00:01, 163.55it/s]     79%|#######9  | 790/1000 [00:01<00:00, 211.14it/s]     86%|########6 | 860/1000 [00:01<00:00, 266.89it/s]     93%|#########3| 931/1000 [00:01<00:00, 328.29it/s]    100%|##########| 1000/1000 [00:01<00:00, 574.27it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     37%|###7      | 149/400 [00:00<00:00, 1483.26it/s]     82%|########1 | 326/400 [00:00<00:00, 1558.70it/s]    100%|##########| 400/400 [00:00<00:00, 1650.03it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:10,  5.66it/s]     42%|####2     | 170/400 [00:00<00:28,  8.07it/s]     82%|########2 | 329/400 [00:00<00:06, 11.50it/s]    100%|##########| 400/400 [00:00<00:00, 943.94it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     38%|###8      | 152/400 [00:00<00:00, 1518.83it/s]     81%|########  | 323/400 [00:00<00:00, 1570.09it/s]    100%|##########| 400/400 [00:00<00:00, 1634.13it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:14,  5.33it/s]     41%|####      | 163/400 [00:00<00:31,  7.61it/s]     81%|########1 | 324/400 [00:00<00:07, 10.85it/s]    100%|##########| 400/400 [00:00<00:00, 921.26it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     37%|###7      | 149/400 [00:00<00:00, 1483.01it/s]     79%|#######8  | 315/400 [00:00<00:00, 1530.65it/s]    100%|##########| 400/400 [00:00<00:00, 1588.69it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:10,  5.64it/s]     42%|####2     | 168/400 [00:00<00:28,  8.05it/s]     83%|########2 | 332/400 [00:00<00:05, 11.48it/s]    100%|##########| 400/400 [00:00<00:00, 955.97it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     16%|#5        | 159/1000 [00:00<00:00, 1581.06it/s]     34%|###4      | 340/1000 [00:00<00:00, 1642.55it/s]     52%|#####1    | 515/1000 [00:00<00:00, 1670.75it/s]     68%|######8   | 681/1000 [00:00<00:00, 1666.51it/s]     85%|########5 | 853/1000 [00:00<00:00, 1679.76it/s]    100%|##########| 1000/1000 [00:00<00:00, 1702.86it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:28,  4.80it/s]     17%|#6        | 169/1000 [00:00<02:01,  6.85it/s]     33%|###3      | 330/1000 [00:00<01:08,  9.77it/s]     51%|#####     | 509/1000 [00:00<00:35, 13.93it/s]     68%|######7   | 677/1000 [00:00<00:16, 19.82it/s]     84%|########3 | 838/1000 [00:00<00:05, 28.17it/s]    100%|##########| 1000/1000 [00:00<00:00, 1236.65it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     11%|#         | 107/1000 [00:00<00:00, 1062.54it/s]     23%|##3       | 233/1000 [00:00<00:00, 1113.36it/s]     36%|###5      | 356/1000 [00:00<00:00, 1144.47it/s]     48%|####8     | 481/1000 [00:00<00:00, 1173.29it/s]     60%|######    | 603/1000 [00:00<00:00, 1186.03it/s]     73%|#######3  | 730/1000 [00:00<00:00, 1209.32it/s]     86%|########5 | 856/1000 [00:00<00:00, 1222.71it/s]     98%|#########7| 979/1000 [00:00<00:00, 1221.70it/s]    100%|##########| 1000/1000 [00:00<00:00, 1216.92it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:01,  5.52it/s]     16%|#5        | 155/1000 [00:00<01:47,  7.87it/s]     31%|###1      | 314/1000 [00:00<01:01, 11.22it/s]     48%|####8     | 480/1000 [00:00<00:32, 15.98it/s]     64%|######4   | 641/1000 [00:00<00:15, 22.73it/s]     80%|########  | 801/1000 [00:00<00:06, 32.27it/s]     96%|#########5| 955/1000 [00:00<00:00, 45.69it/s]    100%|##########| 1000/1000 [00:00<00:00, 1235.24it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     11%|#         | 108/1000 [00:00<00:00, 1077.70it/s]     24%|##3       | 235/1000 [00:00<00:00, 1128.58it/s]     36%|###6      | 362/1000 [00:00<00:00, 1167.42it/s]     48%|####8     | 484/1000 [00:00<00:00, 1180.35it/s]     61%|######1   | 610/1000 [00:00<00:00, 1202.99it/s]     74%|#######3  | 738/1000 [00:00<00:00, 1222.53it/s]     86%|########6 | 865/1000 [00:00<00:00, 1234.67it/s]    100%|#########9| 996/1000 [00:00<00:00, 1254.56it/s]    100%|##########| 1000/1000 [00:00<00:00, 1240.52it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:00,  5.54it/s]     17%|#7        | 173/1000 [00:00<01:44,  7.90it/s]     34%|###3      | 338/1000 [00:00<00:58, 11.26it/s]     50%|#####     | 500/1000 [00:00<00:31, 16.04it/s]     67%|######6   | 667/1000 [00:00<00:14, 22.81it/s]     83%|########2 | 828/1000 [00:00<00:05, 32.39it/s]     99%|#########9| 994/1000 [00:00<00:00, 45.89it/s]    100%|##########| 1000/1000 [00:00<00:00, 1271.36it/s]
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

        tf_loss = tf.reduce_mean(
            (tf.matmul(tf_input, tf_W) + tf_b - tf_output) ** 2
        )
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
                sess.run(
                    [tf_loss, train_op], feed_dict={tf_input: x, tf_output: y}
                )[0]
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
        value = symjax.nn.schedules.ExponentialMovingAverage(
            x, 0.9, debias=debias
        )[0]
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
    plt.plot(
        SJ_EMA(sample), c="green", label="SJ ema (biased)", linewidth=2, alpha=0.5
    )
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


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  19.154 seconds)


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
