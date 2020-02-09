import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import numpy as np
import matplotlib.pyplot as plt

###### DERIVATIVE OF GAUSSIAN EXAMPLE




#t = T.Placeholder((1000,), 'float32')
#f = T.exp(-(t**2))
#u = f.sum()
#g = theanoxla.gradients(u, [t])
#g2 = theanoxla.gradients(g[0].sum(), [t])
#g3 = theanoxla.gradients(g2[0].sum(), [t])
#
#dog = theanoxla.function(t, outputs=[g[0], g2[0], g3[0]])
#
#plt.plot(np.array(dog(np.linspace(-10, 10, 1000))).T)



###### GRADIENT DESCENT
z = T.Variable(3.)
loss = z**2
g_z = theanoxla.gradients(loss, [z])
print(loss, z)
train = theanoxla.function(outputs=[loss, z], updates={z:z-0.1 * g_z[0]})

losses = list()
values = list()
for i in range(5):
    a, b = train()
    losses.append(a)
    values.append(b)

plt.figure()
plt.subplot(121)
plt.plot(losses)
plt.subplot(122)
plt.plot(values, np.zeros_like(values), 'kx')


###### NOISY GRADIENT DESCENT
z = T.Variable(3.)
loss = z**2 + T.random.randn(())*10
g_z = theanoxla.gradients(loss, [z])
print(loss, g_z)
train = theanoxla.function(outputs=[loss, z], updates={z:z-0.1 * g_z[0]})

losses = list()
values = list()
for i in range(10):
    a, b = train()
    losses.append(a)
    values.append(b)

plt.figure()
plt.subplot(121)
plt.plot(losses)
plt.subplot(122)
plt.plot(values, np.zeros_like(values), 'kx')


plt.show()
####### jacobians

x, y = T.ones(()), T.ones(())
ZZ= T.stack([x, y])
f = T.stack([3*ZZ[0] + 2*ZZ[1]])
j = theanoxla.jacobians(f, [ZZ])[0]
g_j = theanoxla.function(outputs=[j])
print(g_j())

R = T.random.randn()
print('zz', ZZ.roots, (ZZ*10).roots, (ZZ*10*R).roots)
u = ZZ * 10 * R
f = ZZ * 10 * R
print('furoots', f, f.roots, u.roots)
print('\n\n')
j = theanoxla.jacobians(f, [ZZ])[0]
print('\n\n')
print('jroots', j.roots)
g_j = theanoxla.function(outputs=[j])
for i in range(5):
    print(g_j())


plt.show()
asdf


SHAPE = (2, 2)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
z2 = T.Variable(np.ones(SHAPE).astype('float32'), name='z2')

cost = T.sum(T.poolNd(T.cos(z + z2), (1, 1)))
grads = theanoxla.gradients(cost, [z])
p



SHAPE = (2, 2)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
z2 = T.Variable(np.ones(SHAPE).astype('float32'), name='z2')

cost = T.sum(T.poolNd(T.cos(z + z2), (1, 1)))
grads = theanoxla.gradients(cost, [z])
print('gradients', grads)
sgd = theanoxla.optimizers.SGD(grads, [z], 0.001)
adam = theanoxla.optimizers.Adam(grads, [z], 0.001)
print(sgd.updates)
#getgrad = theanoxla.function(outputs=[grads[0]], updates={z2:z2+1})
trainsgd = theanoxla.function(outputs=[cost], updates=sgd.updates)
trainadam = theanoxla.function(outputs=[cost], updates=adam.updates)


#for i in range(10):
#    print(getgrad())
#exit()

cost = list()
for i in range(10):
    cost.append(trainsgd()[0])
    print(z.get({}))

z.reset()

for i in range(10):
    cost.append(trainadam()[0])



import matplotlib.pyplot as plt
plt.plot(cost)
plt.title('cost')
plt.show(block=True)
