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
    [Variable(name=num_steps, shape=(), dtype=int32, trainable=False, scope=/ExponentialMovingAverage/), Placeholder(name=x, shape=(), dtype=float32, scope=/), Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/)]
    Placeholder(name=x, shape=(), dtype=float32, scope=/) Op(name=add, fn=add, shape=(), dtype=float32, scope=/ExponentialMovingAverage/)
    [Variable(name=EMA, shape=(), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/), Placeholder(name=x, shape=(), dtype=float32, scope=/)]
      0%|          | 0/400 [00:00<?, ?it/s]     18%|#7        | 71/400 [00:00<00:00, 709.06it/s]     36%|###5      | 143/400 [00:00<00:00, 706.89it/s]     55%|#####5    | 221/400 [00:00<00:00, 724.76it/s]     70%|######9   | 278/400 [00:00<00:00, 669.67it/s]     86%|########6 | 345/400 [00:00<00:00, 667.82it/s]    100%|##########| 400/400 [00:00<00:00, 692.43it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:51,  3.59it/s]     13%|#3        | 53/400 [00:00<01:07,  5.12it/s]     28%|##8       | 114/400 [00:00<00:39,  7.28it/s]     42%|####2     | 169/400 [00:00<00:22, 10.34it/s]     56%|#####6    | 224/400 [00:00<00:12, 14.66it/s]     70%|#######   | 280/400 [00:00<00:05, 20.71it/s]     85%|########5 | 341/400 [00:00<00:02, 29.16it/s]    100%|##########| 400/400 [00:00<00:00, 409.20it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     24%|##3       | 95/400 [00:00<00:00, 943.93it/s]     50%|####9     | 198/400 [00:00<00:00, 967.99it/s]     78%|#######7  | 310/400 [00:00<00:00, 1008.80it/s]    100%|##########| 400/400 [00:00<00:00, 1045.73it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:31,  4.35it/s]     17%|#7        | 68/400 [00:00<00:53,  6.20it/s]     35%|###5      | 140/400 [00:00<00:29,  8.82it/s]     53%|#####3    | 212/400 [00:00<00:15, 12.53it/s]     70%|#######   | 282/400 [00:00<00:06, 17.76it/s]     88%|########8 | 353/400 [00:00<00:01, 25.10it/s]    100%|##########| 400/400 [00:00<00:00, 499.52it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     26%|##5       | 102/400 [00:00<00:00, 1013.29it/s]     56%|#####6    | 226/400 [00:00<00:00, 1069.87it/s]     86%|########6 | 344/400 [00:00<00:00, 1098.09it/s]    100%|##########| 400/400 [00:00<00:00, 1146.90it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:28,  4.51it/s]     18%|#8        | 72/400 [00:00<00:51,  6.42it/s]     36%|###6      | 144/400 [00:00<00:28,  9.13it/s]     54%|#####3    | 215/400 [00:00<00:14, 12.98it/s]     70%|#######   | 281/400 [00:00<00:06, 18.38it/s]     86%|########6 | 346/400 [00:00<00:02, 25.94it/s]    100%|##########| 400/400 [00:00<00:00, 498.68it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     12%|#1        | 117/1000 [00:00<00:00, 1163.48it/s]     26%|##5       | 256/1000 [00:00<00:00, 1223.01it/s]     40%|###9      | 399/1000 [00:00<00:00, 1277.67it/s]     54%|#####4    | 542/1000 [00:00<00:00, 1317.50it/s]     68%|######8   | 684/1000 [00:00<00:00, 1345.11it/s]     82%|########2 | 821/1000 [00:00<00:00, 1349.58it/s]     96%|#########5| 957/1000 [00:00<00:00, 1349.21it/s]    100%|##########| 1000/1000 [00:00<00:00, 1360.55it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:41,  4.50it/s]      6%|5         | 59/1000 [00:00<02:26,  6.41it/s]     12%|#1        | 118/1000 [00:00<01:36,  9.12it/s]     18%|#7        | 175/1000 [00:00<01:03, 12.94it/s]     24%|##3       | 237/1000 [00:00<00:41, 18.32it/s]     30%|##9       | 299/1000 [00:00<00:27, 25.84it/s]     35%|###4      | 349/1000 [00:00<00:18, 35.88it/s]     40%|###9      | 396/1000 [00:00<00:12, 49.42it/s]     44%|####4     | 443/1000 [00:01<00:08, 67.55it/s]     49%|####8     | 489/1000 [00:01<00:05, 88.43it/s]     53%|#####3    | 531/1000 [00:01<00:04, 113.99it/s]     57%|#####7    | 571/1000 [00:01<00:02, 145.03it/s]     61%|######1   | 612/1000 [00:01<00:02, 179.41it/s]     65%|######5   | 652/1000 [00:01<00:01, 210.51it/s]     69%|######9   | 691/1000 [00:01<00:01, 236.00it/s]     73%|#######2  | 728/1000 [00:01<00:01, 262.53it/s]     77%|#######7  | 772/1000 [00:01<00:00, 297.63it/s]     81%|########1 | 811/1000 [00:02<00:00, 320.27it/s]     86%|########5 | 855/1000 [00:02<00:00, 348.25it/s]     90%|######### | 905/1000 [00:02<00:00, 382.11it/s]     95%|#########5| 953/1000 [00:02<00:00, 404.69it/s]    100%|##########| 1000/1000 [00:02<00:00, 403.86it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      9%|8         | 88/1000 [00:00<00:01, 876.94it/s]     22%|##2       | 220/1000 [00:00<00:00, 975.12it/s]     36%|###5      | 358/1000 [00:00<00:00, 1068.47it/s]     50%|#####     | 505/1000 [00:00<00:00, 1162.76it/s]     66%|######5   | 656/1000 [00:00<00:00, 1247.38it/s]     81%|########  | 808/1000 [00:00<00:00, 1318.24it/s]     96%|#########6| 963/1000 [00:00<00:00, 1377.89it/s]    100%|##########| 1000/1000 [00:00<00:00, 1375.81it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:51,  4.31it/s]      7%|7         | 71/1000 [00:00<02:31,  6.14it/s]     14%|#4        | 141/1000 [00:00<01:38,  8.74it/s]     21%|##1       | 213/1000 [00:00<01:03, 12.42it/s]     29%|##8       | 286/1000 [00:00<00:40, 17.61it/s]     36%|###5      | 357/1000 [00:00<00:25, 24.89it/s]     42%|####2     | 424/1000 [00:00<00:16, 34.99it/s]     49%|####8     | 486/1000 [00:00<00:10, 48.79it/s]     55%|#####4    | 548/1000 [00:01<00:06, 67.39it/s]     62%|######1   | 617/1000 [00:01<00:04, 92.35it/s]     68%|######8   | 683/1000 [00:01<00:02, 124.44it/s]     75%|#######5  | 752/1000 [00:01<00:01, 164.90it/s]     82%|########1 | 818/1000 [00:01<00:00, 210.43it/s]     89%|########8 | 887/1000 [00:01<00:00, 265.45it/s]     96%|#########5| 959/1000 [00:01<00:00, 327.03it/s]    100%|##########| 1000/1000 [00:01<00:00, 581.14it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     13%|#3        | 131/1000 [00:00<00:00, 1305.36it/s]     29%|##9       | 292/1000 [00:00<00:00, 1383.70it/s]     45%|####5     | 454/1000 [00:00<00:00, 1445.67it/s]     62%|######1   | 617/1000 [00:00<00:00, 1494.51it/s]     78%|#######7  | 779/1000 [00:00<00:00, 1527.61it/s]     94%|#########3| 936/1000 [00:00<00:00, 1538.80it/s]    100%|##########| 1000/1000 [00:00<00:00, 1544.69it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<04:21,  3.83it/s]      6%|5         | 57/1000 [00:00<02:53,  5.45it/s]     11%|#1        | 114/1000 [00:00<01:54,  7.75it/s]     17%|#7        | 174/1000 [00:00<01:14, 11.02it/s]     23%|##3       | 231/1000 [00:00<00:49, 15.61it/s]     29%|##9       | 290/1000 [00:00<00:32, 22.04it/s]     35%|###4      | 349/1000 [00:00<00:21, 30.98it/s]     41%|####      | 407/1000 [00:00<00:13, 43.27it/s]     47%|####7     | 474/1000 [00:01<00:08, 60.13it/s]     54%|#####3    | 539/1000 [00:01<00:05, 82.61it/s]     60%|######    | 603/1000 [00:01<00:03, 111.77it/s]     67%|######6   | 668/1000 [00:01<00:02, 148.56it/s]     74%|#######3  | 737/1000 [00:01<00:01, 194.06it/s]     80%|########  | 804/1000 [00:01<00:00, 246.61it/s]     87%|########7 | 872/1000 [00:01<00:00, 304.49it/s]     94%|#########3| 940/1000 [00:01<00:00, 364.47it/s]    100%|##########| 1000/1000 [00:01<00:00, 536.44it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     38%|###8      | 153/400 [00:00<00:00, 1528.01it/s]     84%|########3 | 334/400 [00:00<00:00, 1602.44it/s]    100%|##########| 400/400 [00:00<00:00, 1679.27it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:11,  5.57it/s]     38%|###7      | 151/400 [00:00<00:31,  7.95it/s]     79%|#######8  | 315/400 [00:00<00:07, 11.33it/s]    100%|##########| 400/400 [00:00<00:00, 928.20it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     38%|###7      | 150/400 [00:00<00:00, 1493.40it/s]     82%|########1 | 327/400 [00:00<00:00, 1566.83it/s]    100%|##########| 400/400 [00:00<00:00, 1638.91it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:22,  4.85it/s]     44%|####4     | 177/400 [00:00<00:32,  6.92it/s]     87%|########7 | 349/400 [00:00<00:05,  9.87it/s]    100%|##########| 400/400 [00:00<00:00, 917.00it/s]
      0%|          | 0/400 [00:00<?, ?it/s]     42%|####1     | 167/400 [00:00<00:00, 1661.87it/s]     90%|######### | 362/400 [00:00<00:00, 1738.45it/s]    100%|##########| 400/400 [00:00<00:00, 1812.94it/s]
      0%|          | 0/400 [00:00<?, ?it/s]      0%|          | 1/400 [00:00<01:08,  5.79it/s]     44%|####4     | 177/400 [00:00<00:26,  8.26it/s]     87%|########7 | 348/400 [00:00<00:04, 11.78it/s]    100%|##########| 400/400 [00:00<00:00, 986.29it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     15%|#5        | 150/1000 [00:00<00:00, 1497.48it/s]     31%|###1      | 314/1000 [00:00<00:00, 1535.71it/s]     49%|####8     | 489/1000 [00:00<00:00, 1592.01it/s]     66%|######6   | 663/1000 [00:00<00:00, 1632.81it/s]     84%|########3 | 838/1000 [00:00<00:00, 1666.19it/s]    100%|##########| 1000/1000 [00:00<00:00, 1682.30it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:55,  5.70it/s]     17%|#6        | 168/1000 [00:00<01:42,  8.13it/s]     33%|###3      | 332/1000 [00:00<00:57, 11.59it/s]     50%|####9     | 498/1000 [00:00<00:30, 16.51it/s]     66%|######6   | 661/1000 [00:00<00:14, 23.49it/s]     82%|########2 | 824/1000 [00:00<00:05, 33.35it/s]     98%|#########8| 985/1000 [00:00<00:00, 47.22it/s]    100%|##########| 1000/1000 [00:00<00:00, 1271.10it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     15%|#5        | 151/1000 [00:00<00:00, 1507.88it/s]     33%|###2      | 329/1000 [00:00<00:00, 1578.10it/s]     50%|#####     | 505/1000 [00:00<00:00, 1628.39it/s]     68%|######8   | 685/1000 [00:00<00:00, 1674.63it/s]     86%|########5 | 859/1000 [00:00<00:00, 1693.68it/s]    100%|##########| 1000/1000 [00:00<00:00, 1723.79it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<02:57,  5.62it/s]     17%|#6        | 167/1000 [00:00<01:43,  8.02it/s]     33%|###2      | 329/1000 [00:00<00:58, 11.44it/s]     47%|####6     | 469/1000 [00:00<00:32, 16.28it/s]     62%|######2   | 620/1000 [00:00<00:16, 23.15it/s]     79%|#######8  | 789/1000 [00:00<00:06, 32.88it/s]     95%|#########5| 952/1000 [00:00<00:01, 46.56it/s]    100%|##########| 1000/1000 [00:00<00:00, 1230.93it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]     11%|#1        | 111/1000 [00:00<00:00, 1109.26it/s]     21%|##1       | 212/1000 [00:00<00:00, 1074.86it/s]     31%|###       | 307/1000 [00:00<00:00, 1032.80it/s]     40%|####      | 402/1000 [00:00<00:00, 1002.72it/s]     51%|#####     | 509/1000 [00:00<00:00, 1020.92it/s]     61%|######1   | 613/1000 [00:00<00:00, 1024.57it/s]     70%|#######   | 704/1000 [00:00<00:00, 987.14it/s]      80%|########  | 805/1000 [00:00<00:00, 991.88it/s]     91%|######### | 908/1000 [00:00<00:00, 1001.91it/s]    100%|##########| 1000/1000 [00:00<00:00, 1018.65it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]      0%|          | 1/1000 [00:00<03:08,  5.30it/s]     16%|#6        | 164/1000 [00:00<01:50,  7.56it/s]     31%|###       | 309/1000 [00:00<01:04, 10.78it/s]     47%|####6     | 469/1000 [00:00<00:34, 15.36it/s]     63%|######2   | 627/1000 [00:00<00:17, 21.85it/s]     79%|#######9  | 791/1000 [00:00<00:06, 31.03it/s]     96%|#########5| 957/1000 [00:00<00:00, 43.98it/s]    100%|##########| 1000/1000 [00:00<00:00, 1225.35it/s]
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

   **Total running time of the script:** ( 0 minutes  20.414 seconds)


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
