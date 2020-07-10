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
    Dataset mnist loaded in 0.80s.
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
    Train Loss and Accu [3.4418683 0.46875  ]
    Test Loss and Accu: [1.73178   0.5822316]
    Train Loss and Accu [0.7434195 0.7883065]
    Test Loss and Accu: [0.61963654 0.7948718 ]
    Train Loss and Accu [0.4261073 0.8770161]
    Test Loss and Accu: [0.37661102 0.8838141 ]
    Train Loss and Accu [0.32130373 0.9107863 ]
    Test Loss and Accu: [0.3132592 0.9092548]
    Train Loss and Accu [0.26758078 0.92741936]
    Test Loss and Accu: [0.26993492 0.9197716 ]
    Train Loss and Accu [0.21081425 0.9395161 ]
    Test Loss and Accu: [0.2637209  0.92327726]
    Train Loss and Accu [0.17054023 0.95060486]
    Test Loss and Accu: [0.2689621  0.92007214]
    Train Loss and Accu [0.14728892 0.9621976 ]
    Test Loss and Accu: [0.2956763  0.90494794]
    Train Loss and Accu [0.12621711 0.96622986]
    Test Loss and Accu: [0.20010045 0.93940306]
    Train Loss and Accu [0.10153283 0.97278225]

    Text(0.5, 1.0, 'MNIST (1K data) classification task')





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
        print("Train Loss and Accu", np.mean(L, 0))

    plt.plot(test_accuracy)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("MNIST (1K data) classification task")


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 1 minutes  3.856 seconds)


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
