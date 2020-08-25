.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_timing_adam.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_timing_adam.py:


Computation times
=================

In this example we demonstrate how to perform a simple optimization with Adam
in TF and SymJAX and compare the computation time



.. image:: /auto_examples/images/sphx_glr_plot_timing_adam_001.svg
    :alt: plot timing adam
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    False 10
    TF1
    SJ
    False 100
    TF1
    SJ
    False 200
    TF1
    SJ
    False 400
    TF1
    SJ
    False 1000
    TF1
    SJ
    True 10
    TF1
    SJ
    True 100
    TF1
    SJ
    True 200
    TF1
    SJ
    True 400
    TF1
    SJ
    True 1000
    TF1
    SJ
    /home/vrael/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())






|


.. code-block:: default


    import matplotlib.pyplot as plt

    import symjax
    import symjax.tensor as T
    from symjax.nn import optimizers
    import numpy as np
    import time


    lr = 0.01
    BS = 10000
    D = 1000
    X = np.random.randn(BS, D).astype("float32")
    Y = X.dot(np.random.randn(D, 1).astype("float32")) + 2


    def TF1(x, y, N, preallocate=False):
        import tensorflow.compat.v1 as tf

        tf.compat.v1.disable_v2_behavior()

        if preallocate:
            tf_input = tf.constant(x)
            tf_output = tf.constant(y)
        else:
            tf_input = tf.placeholder(dtype=tf.float32, shape=[BS, D])
            tf_output = tf.placeholder(dtype=tf.float32, shape=[BS, 1])

        np.random.seed(0)

        tf_W = tf.Variable(np.random.randn(D, 1).astype("float32"))
        tf_b = tf.Variable(np.random.randn(1,).astype("float32"))

        tf_loss = tf.reduce_mean(
            (tf.matmul(tf_input, tf_W) + tf_b - tf_output) ** 2
        )

        train_op = tf.train.AdamOptimizer(lr).minimize(tf_loss)

        # initialize session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())

        if not preallocate:
            t = time.time()
            for i in range(N):
                sess.run(train_op, feed_dict={tf_input: x, tf_output: y})
        else:
            t = time.time()
            for i in range(N):
                sess.run(train_op)
        return time.time() - t


    def TF2(x, y, N, preallocate=False):
        import tensorflow as tf

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        np.random.seed(0)

        tf_W = tf.Variable(np.random.randn(D, 1).astype("float32"))
        tf_b = tf.Variable(np.random.randn(1,).astype("float32"))

        @tf.function
        def train(tf_input, tf_output):

            with tf.GradientTape() as tape:
                tf_loss = tf.reduce_mean(
                    (tf.matmul(tf_input, tf_W) + tf_b - tf_output) ** 2
                )
            grads = tape.gradient(tf_loss, [tf_W, tf_b])
            optimizer.apply_gradients(zip(grads, [tf_W, tf_b]))
            return tf_loss

        if preallocate:
            x = tf.constant(x)
            y = tf.constant(y)

        t = time.time()
        for i in range(N):
            l = train(x, y)

        return time.time() - t


    def SJ(x, y, N, preallocate=False):
        symjax.current_graph().reset()
        sj_input = T.Placeholder(dtype=np.float32, shape=[BS, D])
        sj_output = T.Placeholder(dtype=np.float32, shape=[BS, 1])

        np.random.seed(0)

        sj_W = T.Variable(np.random.randn(D, 1).astype("float32"))
        sj_b = T.Variable(np.random.randn(1,).astype("float32"))

        sj_loss = ((sj_input.dot(sj_W) + sj_b - sj_output) ** 2).mean()

        optimizers.Adam(sj_loss, lr)

        train = symjax.function(sj_input, sj_output, updates=symjax.get_updates())

        if preallocate:
            import jax

            x = jax.device_put(x)
            y = jax.device_put(y)

        t = time.time()
        for i in range(N):
            train(x, y)

        return time.time() - t


    values = []
    Ns = [10, 100, 200, 400, 1000]
    for pre in [False, True]:
        for N in Ns:
            print(pre, N)
            print("TF1")
            values.append(TF1(X, Y, N, pre))
            # print("TF2")
            # values.append(TF2(X, Y, N, pre))
            print("SJ")
            values.append(SJ(X, Y, N, pre))


    values = np.array(values).reshape((2, len(Ns), 2))

    for i, ls in enumerate(["-", "--"]):
        for j, c in enumerate(["r", "g"]):
            plt.plot(
                Ns, values[i, :, j], linestyle=ls, c=c, linewidth=3, alpha=0.8
            )
    plt.legend(
        ["TF1 no prealloc.", "SJ no prealloc.", "TF1 prealloc.", "SJ prealloc."]
    )
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  17.226 seconds)


.. _sphx_glr_download_auto_examples_plot_timing_adam.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_timing_adam.py <plot_timing_adam.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_timing_adam.ipynb <plot_timing_adam.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
