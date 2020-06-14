.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_examples_plot_cifar10_classif.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_examples_plot_cifar10_classif.py:


CIFAR10 classification
======================

example of image classification



.. image:: /auto_examples/images/sphx_glr_plot_cifar10_classif_001.svg
    :alt: CIFAR10 classification task
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Loading cifar10:   0%|          | 0/5 [00:00<?, ?it/s]    Loading cifar10:  20%|##        | 1/5 [00:02<00:08,  2.14s/it]    Loading cifar10:  40%|####      | 2/5 [00:02<00:05,  1.72s/it]    Loading cifar10:  60%|######    | 3/5 [00:03<00:02,  1.37s/it]    Loading cifar10:  80%|########  | 4/5 [00:03<00:01,  1.01s/it]    Loading cifar10: 100%|##########| 5/5 [00:04<00:00,  1.08it/s]    Loading cifar10: 100%|##########| 5/5 [00:04<00:00,  1.15it/s]
    Dataset cifar10 loaded in4.84s.
    (32, 3, 3, 3)
    (32, 32, 3, 3)
    (32, 32, 3, 3)
    (32, 32, 3, 3)
    (32, 32, 3, 3)
    (32, 32, 3, 3)
    (32, 32, 3, 3)
    (32, 32, 3, 3)
    (32, 10)
    (32, 3, 32, 32)
    (32, 32, 32, 32)
    (32, 32, 32, 32)
    (32, 32, 32, 32)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 16, 16)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 8, 8)
    (32, 32, 4, 4)
    (32, 32, 4, 4)
    (32, 32, 4, 4)
    (32, 32, 4, 4)
    (32, 32, 1, 1)
    (32, 10)
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W, shape=(32, 3, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_500, shape=(32, 3, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_1, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_516, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_532, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_2, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_548, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_3, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_564, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_1, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_580, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_4, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_596, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_5, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_612, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_2, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_628, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_6, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_644, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_7, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_660, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_3, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_676, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_8, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_692, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_9, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_708, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_4, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_724, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_10, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_740, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_11, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_756, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_5, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_772, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_12, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_788, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_13, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_804, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_6, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_820, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_14, shape=(32, 32, 3, 3), dtype=None, trainable=True, scope=/) Op(name=unnamed_836, shape=(32, 32, 3, 3), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_15, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_852, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_7, shape=(1, 32, 1, 1), dtype=None, trainable=True, scope=/) Op(name=unnamed_868, shape=(1, 32, 1, 1), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=W_16, shape=(32, 10), dtype=None, trainable=True, scope=/) Op(name=unnamed_884, shape=(32, 10), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=b_8, shape=(10,), dtype=None, trainable=True, scope=/) Op(name=unnamed_900, shape=(10,), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    /home/vrael/SymJAX/symjax/base.py:441: UserWarning: Variable and update Variable(name=step_8, shape=(1,), dtype=None, trainable=False, scope=/) Op(name=unnamed_901, shape=(1,), dtype=float32, scope=/)are not the same dtype... attempting to cast
      "are not the same dtype... attempting to cast")
    Test Loss and Accu: [2.3417988  0.09995993]
    Train Loss and Accu [1.4994501 0.4457026]
    Test Loss and Accu: [1.5779954  0.48036858]
    Train Loss and Accu [1.1472725 0.5916093]
    Test Loss and Accu: [1.1623799 0.5942508]
    Train Loss and Accu [1.0115813  0.64224553]

    Text(0.5, 1.0, 'CIFAR10 classification task')





|


.. code-block:: default

    import symjax.tensor as T
    import symjax as sj
    import numpy as np
    import matplotlib.pyplot as plt


    # load the dataset
    cifar10 = sj.data.cifar10.load()

    # some renormalization
    cifar10['train_set/images'] /= cifar10['train_set/images'].max((1, 2, 3), keepdims=True)
    cifar10['test_set/images'] /= cifar10['test_set/images'].max((1, 2, 3), keepdims=True)

    # create the network
    BATCH_SIZE = 32
    images = T.Placeholder((BATCH_SIZE,3, 32, 32), 'float32')
    labels = T.Placeholder((BATCH_SIZE,), 'int32')
    deterministic = T.Placeholder((1,), 'bool')

    layer = [sj.layers.RandomCrop(images, crop_shape=(3, 32, 32),
                    padding=[(0, 0), (4, 4), (4, 4)],
                    deterministic=deterministic)]

    for l in range(8):
        layer.append(sj.layers.Conv2D(layer[-1], 32, (3, 3), b=None, pad='SAME'))
        layer.append(sj.layers.BatchNormalization(layer[-1], [0, 2, 3],
                                        deterministic))
        layer.append(sj.layers.Lambda(layer[-1], T.leaky_relu))
        if l % 3 == 0:
            layer.append(sj.layers.Pool2D(layer[-1], (2, 2)))

    layer.append(sj.layers.Pool2D(layer[-1], layer[-1].shape[2:], pool_type='AVG'))

    layer.append(sj.layers.Dense(layer[-1], 10))

    # each layer is itself a tensor which represents its output and thus
    # any tensor operation can be used on the layer instance, for example
    for l in layer:
        print(l.shape)


    loss = sj.losses.sparse_crossentropy_logits(labels, layer[-1]).mean()
    accuracy = sj.losses.accuracy(labels, layer[-1])

    lr=sj.schedules.PiecewiseConstant(0.01, {15: 0.001, 25: 0.0001})
    opt = sj.optimizers.Adam(loss, lr)

    network_updates = sj.layers.get_updates(layer)

    test = sj.function(images, labels, deterministic, outputs=[loss, accuracy])

    train = sj.function(images, labels, deterministic,
                        outputs=[loss, accuracy], updates={**opt.updates,
                                                    **network_updates})

    test_accuracy = []

    for epoch in range(3):
        L = list()
        for x, y in sj.data.batchify(cifar10['test_set/images'], cifar10['test_set/labels'], batch_size=BATCH_SIZE,
                                          option='continuous'):
            L.append(test(x, y, 1))
        print('Test Loss and Accu:', np.mean(L, 0))
        test_accuracy.append(np.mean(L, 0))
        L = list()
        for x, y in sj.data.batchify(cifar10['train_set/images'], cifar10['train_set/labels'],
                                batch_size=BATCH_SIZE, option='random_see_all'):
            L.append(train(x, y, 0))
        print('Train Loss and Accu', np.mean(L, 0))
        lr.update()

    plt.plot(test_accuracy)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('CIFAR10 classification task')


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 3 minutes  31.527 seconds)


.. _sphx_glr_download_auto_examples_plot_cifar10_classif.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: plot_cifar10_classif.py <plot_cifar10_classif.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: plot_cifar10_classif.ipynb <plot_cifar10_classif.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
