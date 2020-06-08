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

