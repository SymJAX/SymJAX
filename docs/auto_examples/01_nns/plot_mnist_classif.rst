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
    Dataset mnist loaded in 0.93s.
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
    Test Loss and Accu: [2.3026714  0.10276442]
    Train Loss and Accu [1.9802395 0.4092742]
    ...epoch: 1
    Test Loss and Accu: [2.5816264  0.11348157]
    Train Loss and Accu [1.6424525  0.63004035]
    ...epoch: 2
    Test Loss and Accu: [1.9070722  0.20933494]
    Train Loss and Accu [1.4449486  0.70060486]
    ...epoch: 3
    Test Loss and Accu: [1.4708972 0.5521835]
    Train Loss and Accu [1.2993451 0.7641129]
    ...epoch: 4
    Test Loss and Accu: [1.2808527 0.6897035]
    Train Loss and Accu [1.1622738 0.8059476]
    ...epoch: 5
    Test Loss and Accu: [1.1994443 0.7092348]
    Train Loss and Accu [1.0483605 0.828125 ]
    ...epoch: 6
    Test Loss and Accu: [1.0837309 0.7782452]
    Train Loss and Accu [0.9590779 0.8518145]
    ...epoch: 7
    Test Loss and Accu: [1.0044572 0.7897636]
    Train Loss and Accu [0.8599169  0.87247986]
    ...epoch: 8
    Test Loss and Accu: [0.96697557 0.76011616]
    Train Loss and Accu [0.7870807  0.88810486]
    ...epoch: 9
    Test Loss and Accu: [0.9209642  0.75340545]
    Train Loss and Accu [0.7239348 0.8966734]
    ...epoch: 10
    Test Loss and Accu: [0.9281626 0.7411859]
    Train Loss and Accu [0.6565298 0.9128024]
    ...epoch: 11
    Test Loss and Accu: [0.9744095  0.64813703]
    Train Loss and Accu [0.6110032 0.9128024]
    ...epoch: 12
    Test Loss and Accu: [0.8637302 0.7772436]
    Train Loss and Accu [0.5666782 0.9188508]
    ...epoch: 13
    Test Loss and Accu: [0.76896995 0.7823518 ]
    Train Loss and Accu [0.51209766 0.9304435 ]
    ...epoch: 14
    Test Loss and Accu: [0.63755596 0.85396636]
    Train Loss and Accu [0.477681  0.9354839]
    ...epoch: 15
    Test Loss and Accu: [0.83588773 0.72375804]
    Train Loss and Accu [0.44902894 0.94102824]
    ...epoch: 16
    Test Loss and Accu: [0.56575525 0.8772035 ]
    Train Loss and Accu [0.41248325 0.9465726 ]
    ...epoch: 17
    Test Loss and Accu: [0.58106077 0.8576723 ]
    Train Loss and Accu [0.3823703 0.9551411]
    ...epoch: 18
    Test Loss and Accu: [0.61149865 0.8235176 ]
    Train Loss and Accu [0.35944167 0.9611895 ]
    ...epoch: 19
    Test Loss and Accu: [0.6215606 0.8192107]
    Train Loss and Accu [0.34505492 0.95866936]

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

    mnist["train_set/images"] /= mnist["train_set/images"].max(
        (1, 2, 3), keepdims=True
    )
    mnist["test_set/images"] /= mnist["test_set/images"].max(
        (1, 2, 3), keepdims=True
    )

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


    layer.append(
        nn.layers.Pool2D(layer[-1], layer[-1].shape.get()[2:], pool_type="AVG")
    )

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

   **Total running time of the script:** ( 2 minutes  6.271 seconds)


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
