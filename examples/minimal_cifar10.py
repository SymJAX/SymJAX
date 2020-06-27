import sys

sys.path.insert(0, "../")
import symjax.tensor as T
import symjax as sj
import numpy as np

# load the dataset
images_train, labels_train, images_test, labels_test = sj.datasets.cifar10.load()

# some renormalization
images_train /= images_train.max((1, 2, 3), keepdims=True)
images_test /= images_test.max((1, 2, 3), keepdims=True)

# create the network
BATCH_SIZE = 32
inputs = T.Placeholder((BATCH_SIZE,) + images_train.shape[1:], "float32")
outputs = T.Placeholder((BATCH_SIZE,), "int32")
deterministic = T.Placeholder((1,), "bool")

layer = [
    sj.layers.RandomCrop(
        inputs,
        crop_shape=(3, 32, 32),
        padding=[(0, 0), (4, 4), (4, 4)],
        deterministic=deterministic,
    )
]
layer.append(sj.layers.Conv2D(layer[-1], (32, 3, 3, 3)))
layer.append(sj.layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(sj.layers.Activation(layer[-1], T.relu))
layer.append(sj.layers.Pool2D(layer[-1], (2, 2)))

layer.append(sj.layers.Conv2D(layer[-1], (64, 32, 3, 3)))
layer.append(sj.layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(sj.layers.Activation(layer[-1], T.relu))
layer.append(sj.layers.Pool2D(layer[-1], (2, 2)))

layer.append(sj.layers.Dense(layer[-1], 128))
layer.append(sj.layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(sj.layers.Activation(layer[-1], T.relu))
layer.append(sj.layers.Dense(layer[-1], 10))

# each layer is itself a tensor which represents its output and thus
# any tensor operation can be used on the layer instance, for example
for l in layer:
    print(l.shape)

# (32, 3, 32, 32)
# (32, 32, 30, 30)
# (32, 32, 30, 30)
# (32, 32, 30, 30)
# (32, 32, 15, 15)
# (32, 64, 13, 13)
# (32, 64, 13, 13)
# (32, 64, 13, 13)
# (32, 64, 6, 6)
# (32, 128)
# (32, 128)
# (32, 128)
# (32, 10)

loss = sj.losses.sparse_crossentropy_logits(outputs, layer[-1]).mean()
accuracy = sj.losses.accuracy(outputs, layer[-1])

params = sum([lay.variables() for lay in layer], [])

lr = sj.schedules.PiecewiseConstant(0.005, {50: 0.001, 75: 0.0005})
opt = sj.optimizers.Adam(loss, params, lr)

for l in layer:
    opt.updates.update(l.updates)


test = sj.function(inputs, outputs, deterministic, outputs=[loss, accuracy])

train = sj.function(
    inputs, outputs, deterministic, outputs=[loss, accuracy], updates=opt.updates
)

for epoch in range(100):
    L = list()
    for x, y in sj.utils.batchify(
        images_test, labels_test, batch_size=BATCH_SIZE, option="continuous"
    ):
        L.append(test(x, y, 1))
    print("Test Loss and Accu:", np.mean(L, 0))
    L = list()
    for x, y in sj.utils.batchify(
        images_train, labels_train, batch_size=BATCH_SIZE, option="random_see_all"
    ):
        L.append(train(x, y, 0))
    print("Train Loss and Accu", np.mean(L, 0))
    lr.update()

# Test Loss and Accu: [2.6886015  0.09194712]
# Train Loss and Accu [1.3671544  0.51288414]
# Test Loss and Accu: [1.7053369  0.43449518]
# Train Loss and Accu [1.1127299 0.6065541]
# Test Loss and Accu: [1.1878427  0.59094554]
# Train Loss and Accu [1.0067393 0.6460667]
# Test Loss and Accu: [1.1366144 0.6133814]
# Train Loss and Accu [0.9416873  0.66995436]
# Test Loss and Accu: [0.95114607 0.6744792 ]
# Train Loss and Accu [0.891217   0.68737996]
# Test Loss and Accu: [1.272816  0.5885417]
# Train Loss and Accu [0.84912854 0.7034651 ]
# Test Loss and Accu: [0.81524473 0.7214543 ]
# .....
