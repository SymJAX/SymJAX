.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_image_transformation.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_image_transformation.py:


Basic image transform (TPS/affine)
==================================

In this example we demonstrate how to employ the utility functions from
``symjax.tensor.interpolation.affine_transform`` and
``symjax.tensor.interpolation.thin_plate_spline``
to transform/interpolate images



.. image:: /auto_examples/images/sphx_glr_plot_image_transformation_001.svg
    :alt: original, identity, x translation, y translation, random, zoom, zoom, blob, original, identity, x translation, y translation, skewness x, zoom, zoom x, skewness y
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /home/vrael/SymJAX/symjax/tensor/interpolation.py:548: RuntimeWarning: divide by zero encountered in log
      log_r_2 = np.log(r_2)
            ... mnist.pkl.gz already exists
    Loading mnist
    Dataset mnist loaded in 1.07s.
    /home/vrael/anaconda3/lib/python3.7/site-packages/matplotlib/figure.py:445: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
      % get_backend())






|


.. code-block:: default


    import matplotlib.pyplot as plt
    import symjax
    import symjax.tensor as T
    import numpy as np

    x = T.Placeholder((10, 1, 28, 28), "float32")
    points = T.Placeholder((10, 2 * 16), "float32")
    thetas = T.Placeholder((10, 6), "float32")

    affine = T.interpolation.affine_transform(x, thetas)
    tps = T.interpolation.thin_plate_spline(x, points)

    f = symjax.function(x, thetas, outputs=affine)
    g = symjax.function(x, points, outputs=tps)


    data = symjax.data.mnist()["train_set/images"][:10]


    plt.figure(figsize=(20, 6))
    plt.subplot(2, 8, 1)
    plt.imshow(data[0][0])
    plt.title("original")
    plt.ylabel("TPS")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 2)
    points = np.zeros((10, 2 * 16))
    plt.imshow(g(data, points)[0][0])
    plt.title("identity")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 3)
    points = np.zeros((10, 2 * 16))
    points[:, :16] += 0.3
    plt.imshow(g(data, points)[0][0])
    plt.title("x translation")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 4)
    points = np.zeros((10, 2 * 16))
    points[:, 16:] += 0.3
    plt.imshow(g(data, points)[0][0])
    plt.title("y translation")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 5)
    points = np.random.randn(10, 2 * 16) * 0.2
    plt.imshow(g(data, points)[0][0])
    plt.title("random")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 6)
    points = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    points = np.concatenate([points[0].reshape(-1), points[1].reshape(-1)]) * 0.4
    points = points[None] * np.ones((10, 1))
    plt.imshow(g(data, points)[0][0])
    plt.title("zoom")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 7)
    points = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    points = np.concatenate([points[0].reshape(-1), points[1].reshape(-1)]) * -0.2
    points = points[None] * np.ones((10, 1))
    plt.imshow(g(data, points)[0][0])
    plt.title("zoom")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 8)
    points = np.zeros((10, 2 * 16))
    points[:, 1::2] -= 0.1
    points[:, ::2] += 0.1
    plt.imshow(g(data, points)[0][0])
    plt.title("blob")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 9)
    plt.imshow(data[0][0])
    plt.title("original")
    plt.ylabel("Affine")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 10)
    points = np.zeros((10, 6))
    points[:, 0] = 1
    points[:, 4] = 1
    plt.imshow(f(data, points)[0][0])
    plt.title("identity")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 11)
    points = np.zeros((10, 6))
    points[:, 0] = 1
    points[:, 4] = 1
    points[:, 2] = 0.2
    plt.imshow(f(data, points)[0][0])
    plt.title("x translation")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 12)
    points = np.zeros((10, 6))
    points[:, 0] = 1
    points[:, 4] = 1
    points[:, 5] = 0.2
    plt.imshow(f(data, points)[0][0])
    plt.title("y translation")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 13)
    points = np.zeros((10, 6))
    points[:, 0] = 1
    points[:, 4] = 1
    points[:, 1] = 0.4
    plt.imshow(f(data, points)[0][0])
    plt.title("skewness x")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 14)
    points = np.zeros((10, 6))
    points[:, 0] = 1.4
    points[:, 4] = 1.4
    plt.imshow(f(data, points)[0][0])
    plt.title("zoom")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 15)
    points = np.zeros((10, 6))
    points[:, 0] = 1.4
    points[:, 4] = 1.0
    plt.imshow(f(data, points)[0][0])
    plt.title("zoom x")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 8, 16)
    points = np.zeros((10, 6))
    points[:, 0] = 1
    points[:, 4] = 1
    points[:, 3] = 0.4
    plt.imshow(f(data, points)[0][0])
    plt.title("skewness y")
    plt.xticks([])
    plt.yticks([])


    plt.tight_layout()
    plt.show()


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  3.074 seconds)


.. _sphx_glr_download_auto_examples_plot_image_transformation.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_image_transformation.py <plot_image_transformation.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_image_transformation.ipynb <plot_image_transformation.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
