.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_deconvolution.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_deconvolution.py:


Basic (linear) deconvolution filter learning
============================================

demonstration on how to learn a deconvolutional filter
based on some flavors of gradietn descent assuming we know
the true output



.. image:: /auto_examples/images/sphx_glr_plot_deconvolution_001.svg
    :alt: reconstruction (l1), reconstruction (l2), True image, Convolved image
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

            ... mnist.pkl.gz already exists
    Loading mnist
    Dataset mnist loaded in 0.95s.
    /home/vrael/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())






|


.. code-block:: default


    import symjax
    import symjax.tensor as T
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import convolve2d

    import os

    os.environ["DATASET_PATH"] = "/home/vrael/DATASETS/"

    symjax.current_graph().reset()


    true_image = symjax.data.mnist()
    # 2d image
    true_image = true_image["train_set/images"][0, 0]
    true_image /= true_image.max()

    np.random.seed(0)

    noisy_image = convolve2d(true_image, np.random.randn(5, 5) / 5, "same")

    # GRADIENT DESCENT
    filter_1 = T.Variable(np.random.randn(8, 8) / 8, dtype="float32")
    filter_2 = T.Variable(filter_1.value, dtype="float32")

    reconstruction_1 = T.signal.convolve2d(noisy_image, filter_1, "same")
    reconstruction_2 = T.signal.convolve2d(noisy_image, filter_2, "same")

    loss1 = T.abs(reconstruction_1 - true_image).mean()
    loss2 = (T.abs(reconstruction_2 - true_image) ** 2).mean()

    lr = symjax.nn.schedules.PiecewiseConstant(0.05, {5000: 0.01, 8000: 0.005})
    symjax.nn.optimizers.Adam(loss1 + loss2, lr)

    train = symjax.function(outputs=[loss1, loss2], updates=symjax.get_updates())

    rec = symjax.function(outputs=[reconstruction_1, reconstruction_2])

    losses_1 = list()
    losses_2 = list()

    for i in range(10000):
        losses = train()
        losses_1.append(losses[0])
        losses_2.append(losses[1])

    reconstruction_1, reconstruction_2 = rec()


    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.semilogy(losses_1, "-x")
    plt.ylabel("log-loss (l1)")
    plt.xlabel("number of gradient updates")


    plt.subplot(222)
    plt.semilogy(losses_2, "-x")
    plt.ylabel("log-loss (l2)")
    plt.xlabel("number of gradient updates")


    plt.subplot(245)
    plt.imshow(reconstruction_1, aspect="auto", origin="lower", cmap="plasma")
    plt.xticks([])
    plt.yticks([])
    plt.title("reconstruction (l1)")


    plt.subplot(246)
    plt.imshow(reconstruction_2, aspect="auto", origin="lower", cmap="plasma")
    plt.xticks([])
    plt.yticks([])
    plt.title("reconstruction (l2)")


    plt.subplot(247)
    plt.imshow(true_image, aspect="auto", origin="lower", cmap="plasma")
    plt.xticks([])
    plt.yticks([])
    plt.title("True image")

    plt.subplot(248)
    plt.imshow(noisy_image, aspect="auto", origin="lower", cmap="plasma")
    plt.xticks([])
    plt.yticks([])
    plt.title("Convolved image")


    plt.tight_layout()
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  32.168 seconds)


.. _sphx_glr_download_auto_examples_plot_deconvolution.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_deconvolution.py <plot_deconvolution.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_deconvolution.ipynb <plot_deconvolution.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
