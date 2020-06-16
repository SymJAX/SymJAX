import symjax as sj
import symjax.tensor as T

w = T.Variable(1., dtype='float32')
u = T.Placeholder((), 'float32')
out = T.map(lambda a, w, u: (u - w) * a, [T.range(3)], non_sequences=[w, u])
f = sj.function(u, outputs=out, updates={w: w + 1})
print(f(2))
# [0, 1, 2]
print(f(2))
# [0, 0, 0]
print(f(0))
# [0, -3, -6]


w.reset()
out = T.map(lambda a, w, u: w * a * u, [T.range(3)], non_sequences=[w, u])
g = sj.gradients(out.sum(), [w])[0]
f = sj.function(u, outputs=g)

print(f(0))
# 0
print(f(1))
# 3


out = T.map(lambda a, b: a * b, [T.range(3), T.range(3)])
f = sj.function(outputs=out)

print(f())
# [0, 1, 4]


w.reset()
v = T.Placeholder((), 'float32')
out = T.while_loop(lambda i, u: i[0] + u < 5,
                   lambda i: (i[0] + 1., i[0] ** 2), (w, 1.),
                   non_sequences_cond=[v])
f = sj.function(v, outputs=out)
print(f(0))
# 5, 16
print(f(2))
# [3, 4]
