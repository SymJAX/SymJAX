.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_01_nns_plot_mnist_classif.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_01_nns_plot_mnist_classif.py:


MNIST classification
====================

example of image (MNIST) classification on small part of the data
and with a small architecture



.. image:: /auto_examples/01_nns/images/sphx_glr_plot_mnist_classif_001.svg
    :alt: MNIST (1K data) classification task
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Loading mnist
    Dataset mnist loaded in 1.94s.
    /home/vrael/anaconda3/envs/jax/lib/python3.7/site-packages/jax/lib/xla_bridge.py:125: UserWarning: No GPU/TPU found, falling back to CPU.
      warnings.warn('No GPU/TPU found, falling back to CPU.')
    (32, 1, 28, 28)
    (32, 32, 28, 28)
    (32, 32, 28, 28)
    (32, 32, 28, 28)
    (32, 32, 14, 14)
    (32, 32, 14, 14)
    (32, 32, 14, 14)
    (32, 32, 14, 14)
    (32, 32, 7, 7)
    (32, 32, 7, 7)
    (32, 32, 7, 7)
    (32, 32, 7, 7)
    (32, 32, 3, 3)
    (32, 32, 1, 1)
    (32, 10)
    Test Loss and Accu: [2.302591   0.09805689]
    Train Loss and Accu [4.317519  0.4077621]
    Test Loss and Accu: [2.0085776  0.54096556]
    Train Loss and Accu [1.0729111  0.71622986]
    Test Loss and Accu: [0.761555   0.76742786]
    Train Loss and Accu [0.5668073 0.8210685]
    Test Loss and Accu: [0.5893349  0.79977965]
    Train Loss and Accu [0.39215317 0.88508064]
    Test Loss and Accu: [0.36009136 0.9001402 ]
    Train Loss and Accu [0.3160174 0.9077621]
    Test Loss and Accu: [0.40013826 0.8711939 ]
    Train Loss and Accu [0.28734395 0.9128024 ]
    Test Loss and Accu: [0.33410507 0.8999399 ]
    Train Loss and Accu [0.216163  0.9470766]
    Test Loss and Accu: [0.29659674 0.9097556 ]
    Train Loss and Accu [0.19616833 0.95060486]
    Test Loss and Accu: [0.28937742 0.91215944]
    Train Loss and Accu [0.18014076 0.95060486]
    Test Loss and Accu: [0.30587545 0.9082532 ]
    Train Loss and Accu [0.16285737 0.95715725]

    Text(0.5, 0.98, 'MNIST (1K data) classification task')





|


.. code-block:: default

    import symjax.tensor as T
    from symjax import nn
    import symjax
    import numpy as np
    import matplotlib.pyplot as plt
    from symjax.data import mnist
    from symjax.data.utils import batchify

    import os

    os.environ["DATASET_PATH"] = "/home/vrael/DATASETS/"
    symjax.current_graph().reset()
    # load the dataset
    mnist = mnist.load()

    # some renormalization, and we only keep the first 2000 images
    mnist["train_set/images"] = mnist["train_set/images"][:2000]
    mnist["train_set/labels"] = mnist["train_set/labels"][:2000]

    mnist["train_set/images"] /= mnist["train_set/images"].max(
        (1, 2, 3), keepdims=True
    )
    mnist["test_set/images"] /= mnist["test_set/images"].max(
        (1, 2, 3), keepdims=True
    )

    # create the network
    BATCH_SIZE = 32
    images = T.Placeholder((BATCH_SIZE, 1, 28, 28), "float32", name="images")
    labels = T.Placeholder((BATCH_SIZE,), "int32", name="labels")
    deterministic = T.Placeholder((1,), "bool")


    layer = [nn.layers.Identity(images)]

    for l in range(3):
        layer.append(nn.layers.Conv2D(layer[-1], 32, (3, 3), b=None, pad="SAME"))
        layer.append(nn.layers.BatchNormalization(layer[-1], [1], deterministic))
        layer.append(nn.leaky_relu(layer[-1]))
        layer.append(nn.layers.Pool2D(layer[-1], (2, 2)))

    layer.append(nn.layers.Pool2D(layer[-1], layer[-1].shape[2:], pool_type="AVG"))
    layer.append(nn.layers.Dense(layer[-1], 10))

    # each layer is itself a tensor which represents its output and thus
    # any tensor operation can be used on the layer instance, for example
    for l in layer:
        print(l.shape)


    loss = nn.losses.sparse_crossentropy_logits(labels, layer[-1]).mean()
    accuracy = nn.losses.accuracy(labels, layer[-1])

    nn.optimizers.Adam(loss, 0.01)

    test = symjax.function(images, labels, deterministic, outputs=[loss, accuracy])

    train = symjax.function(
        images,
        labels,
        deterministic,
        outputs=[loss, accuracy],
        updates=symjax.get_updates(),
    )

    test_accuracy = []
    train_accuracy = []

    for epoch in range(10):
        L = list()
        for x, y in batchify(
            mnist["test_set/images"],
            mnist["test_set/labels"],
            batch_size=BATCH_SIZE,
            option="continuous",
        ):
            L.append(test(x, y, 1))
        print("Test Loss and Accu:", np.mean(L, 0))
        test_accuracy.append(np.mean(L, 0))
        L = list()
        for x, y in batchify(
            mnist["train_set/images"],
            mnist["train_set/labels"],
            batch_size=BATCH_SIZE,
            option="random_see_all",
        ):
            L.append(train(x, y, 0))
        train_accuracy.append(np.mean(L, 0))
        print("Train Loss and Accu", np.mean(L, 0))

    train_accuracy = np.array(train_accuracy)
    test_accuracy = np.array(test_accuracy)

    plt.subplot(121)
    plt.plot(test_accuracy[:, 1], c="k")
    plt.plot(train_accuracy[:, 1], c="b")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    plt.subplot(122)
    plt.plot(test_accuracy[:, 0], c="k")
    plt.plot(train_accuracy[:, 0], c="b")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    plt.suptitle("MNIST (1K data) classification task")


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  26.475 seconds)


.. _sphx_glr_download_auto_examples_01_nns_plot_mnist_classif.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_mnist_classif.py <plot_mnist_classif.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_mnist_classif.ipynb <plot_mnist_classif.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
