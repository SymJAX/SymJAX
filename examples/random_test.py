import symjax
import symjax.tensor as T

# test the random normal/uniform
SHAPE = (1,)
value = T.Variable(T.ones(SHAPE))
value2 = T.Variable(T.zeros(SHAPE))
with symjax.Graph('part2'):
    value3 = T.Variable(T.zeros(SHAPE))
value4 = T.Variable(T.zeros(SHAPE))

o = value4+value3
p1 = value4 * 4
p2 = value4 * 6


randn = T.random.randn(SHAPE)
rand = T.random.rand(SHAPE)

out1 = randn * value
out2 = out1.clone({randn: rand})# * value
#out3 = out1.clone({randn:T.ones(1)*3})
print(out2.roots)
get_vars = symjax.function(outputs=[out1, out2], updates={value:2+value})
two = symjax.function(rand, outputs=[out2], updates={value:2+value})
print('start')
for i in range(100):
    print(two(1))
#    print(get_vars())
#    print(get_vars())

asdf

# test shuffle
matrix = T.linspace(0, 1, 16).reshape((4, 4))
smatrix = matrix[T.random.permutation(T.range(4))]

get_shuffle = symjax.function(outputs=smatrix)

for i in range(10):
    print(get_shuffle())


asdfsadf

# test the random uniform
SHAPE = (2, 2)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
get_z = symjax.function(outputs=z)

for i in range(10):
    print(get_z())






#print(f_shuffle()[0])
asdasd












w = T.Placeholder(SHAPE, 'float32', name='w')
noise = T.random.uniform(SHAPE, dtype='float32')
y = T.cos(symjax.nn.activations.leaky_relu(z,0.3) + w + noise)
cost = T.pool(y, (2, 2))
cost = T.sum(cost)

grads = symjax.gradients(cost, [w, z], [1])

print(cost.get({w: np.random.randn(*SHAPE)}))
noise.seed = 20
print(cost.get({w: np.random.randn(*SHAPE)}))
noise.seed = 40
print(cost.get({w: np.random.randn(*SHAPE)}))

updates = {z:z-0.01*grads[0]}
fn1 = symjax.function(w, outputs=[cost])
fn2 = symjax.function(w, outputs=[cost], updates=updates)
print(fn1(np.random.randn(*SHAPE)))
print(fn1(np.random.randn(*SHAPE)))

cost = list()
for i in range(1000):
    cost.append(fn2(np.ones(SHAPE))[0])

import matplotlib.pyplot as plt
plt.plot(cost)
plt.show()
