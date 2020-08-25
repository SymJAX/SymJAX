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

            ... mnist.pkl.gz already exists
    Loading mnist
    Dataset mnist loaded in 0.89s.
    (64, 1, 28, 28)
    (64, 64, 26, 26)
    (64, 64, 26, 26)
    (64, 64, 26, 26)
    (64, 64, 13, 13)
    (64, 64, 11, 11)
    (64, 64, 11, 11)
    (64, 64, 11, 11)
    (64, 64, 5, 5)
    (64, 64, 1, 1)
    (64, 10)
    ...epoch: 0
    Test Loss and Accu: [2.303416   0.09815705]
    Train Loss and Accu [1.9887893 0.4032258]
    ...epoch: 1
    Test Loss and Accu: [2.8440666  0.11348157]
    Train Loss and Accu [1.6667411  0.59727824]
    ...epoch: 2
    Test Loss and Accu: [1.9810963 0.2217548]
    Train Loss and Accu [1.4739343 0.7021169]
    ...epoch: 3
    Test Loss and Accu: [1.4481678  0.62369794]
    Train Loss and Accu [1.3121623 0.7636089]
    ...epoch: 4
    Test Loss and Accu: [1.3252217 0.6608574]
    Train Loss and Accu [1.1772408  0.81602824]
    ...epoch: 5
    Test Loss and Accu: [1.1886992 0.7281651]
    Train Loss and Accu [1.0639273 0.8251008]
    ...epoch: 6
    Test Loss and Accu: [1.0929953  0.79547274]
    Train Loss and Accu [0.9512595  0.86441535]
    ...epoch: 7
    Test Loss and Accu: [1.0527598 0.7073317]
    Train Loss and Accu [0.87056917 0.88508064]
    ...epoch: 8
    Test Loss and Accu: [0.9697831 0.7624199]
    Train Loss and Accu [0.7969418 0.8840726]
    ...epoch: 9
    Test Loss and Accu: [0.9283603 0.7219551]
    Train Loss and Accu [0.7198963 0.8986895]
    ...epoch: 10
    Test Loss and Accu: [0.8994963  0.73297274]
    Train Loss and Accu [0.66082746 0.9153226 ]
    ...epoch: 11
    Test Loss and Accu: [0.7540454  0.82471955]
    Train Loss and Accu [0.6040695 0.9203629]
    ...epoch: 12
    Test Loss and Accu: [0.6891631  0.86588544]
    Train Loss and Accu [0.55794346 0.9279234 ]
    ...epoch: 13
    Test Loss and Accu: [0.88701797 0.7079327 ]
    Train Loss and Accu [0.5097796 0.9309476]
    ...epoch: 14
    Test Loss and Accu: [0.6690547 0.8370393]
    Train Loss and Accu [0.45673898 0.9485887 ]
    ...epoch: 15
    Test Loss and Accu: [0.6740171  0.82261616]
    Train Loss and Accu [0.42886272 0.94758064]
    ...epoch: 16
    Test Loss and Accu: [0.66504765 0.79917866]
    Train Loss and Accu [0.41574922 0.9465726 ]
    ...epoch: 17
    Test Loss and Accu: [0.5708862 0.8608774]
    Train Loss and Accu [0.36107954 0.9596774 ]
    ...epoch: 18
    Test Loss and Accu: [0.49747434 0.89663464]
    Train Loss and Accu [0.34122872 0.9621976 ]
    ...epoch: 19
    Test Loss and Accu: [0.5516419  0.85917467]
    Train Loss and Accu [0.31652793 0.96622986]

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
    mnist = mnist()

    # some renormalization, and we only keep the first 2000 images
    mnist["train_set/images"] = mnist["train_set/images"][:2000]
    mnist["train_set/labels"] = mnist["train_set/labels"][:2000]

    mnist["train_set/images"] /= mnist["train_set/images"].max((1, 2, 3), keepdims=True)
    mnist["test_set/images"] /= mnist["test_set/images"].max((1, 2, 3), keepdims=True)

    # create the network
    BATCH_SIZE = 64
    images = T.Placeholder((BATCH_SIZE, 1, 28, 28), "float32", name="images")
    labels = T.Placeholder((BATCH_SIZE,), "int32", name="labels")
    deterministic = T.Placeholder((1,), "bool")


    layer = [nn.layers.Identity(images)]

    for l in range(2):
        layer.append(nn.layers.Conv2D(layer[-1], 64, (3, 3), b=None))
        # due to the small size of the dataset we can
        # increase the update of the bn moving averages
        layer.append(
            nn.layers.BatchNormalization(
                layer[-1], [1], deterministic, beta_1=0.9, beta_2=0.9
            )
        )
        layer.append(nn.leaky_relu(layer[-1]))
        layer.append(nn.layers.Pool2D(layer[-1], (2, 2)))


    layer.append(nn.layers.Pool2D(layer[-1], layer[-1].shape.get()[2:], pool_type="AVG"))

    layer.append(nn.layers.Dense(layer[-1], 10))

    # each layer is itself a tensor which represents its output and thus
    # any tensor operation can be used on the layer instance, for example
    for l in layer:
        print(l.shape.get())


    loss = nn.losses.sparse_softmax_crossentropy_logits(labels, layer[-1]).mean()
    accuracy = nn.losses.accuracy(labels, layer[-1])

    nn.optimizers.Adam(loss, 0.001)

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

    for epoch in range(20):
        print("...epoch:", epoch)
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

   **Total running time of the script:** ( 2 minutes  12.597 seconds)


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
