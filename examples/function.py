import symjax
import symjax.tensor as T



value = T.Variable(T.ones(()))
randn = T.random.randn(())
rand = T.random.rand(())

out1 = randn * value
out2 = out1.clone({randn: rand})

f = symjax.function(rand, outputs=out2, updates={value:2+value})

for i in range(3):
    print(f(i))
# 0.
# 3.
# 10.


# we create a simple computational graph
var = T.Variable(T.random.randn((16, 8), seed=10))
loss = ((var - T.ones_like(var))**2 ).sum()
g = symjax.gradients(loss, [var])
opt = symjax.optimizers.SGD(loss, 0.01, params=var)

f = symjax.function(outputs=loss, updates=opt.updates)

for i in range(10):
    print(f())
# 240.96829
# 231.42595
# 222.26149
# 213.45993
# 205.00691
# 196.88864
# 189.09186
# 181.60382
# 174.41231
# 167.50558


