import sys

sys.path.insert(0, "../")
import symjax
import symjax.tensor as T

# create our variable to be optimized
mu = T.Variable(T.random.normal((1,), seed=1))
cost = T.exp(-((mu - 1) ** 2))
lr = symjax.schedules.PiecewiseConstant(0.01, {100: 0.003, 150: 0.001})
opt = symjax.optimizers.Adam(cost, lr, params=[mu])
print(opt.updates)
f = symjax.function(outputs=cost, updates=opt.updates)

for k in range(4):
    for i in range(10):
        print(f())
    print("done")
    for v in opt.variables + [mu]:
        v.reset()
    lr.reset()

# 0.008471076
# 0.008201109
# 0.007946267
# 0.007705368
# 0.0074773384
# 0.007261208
# 0.0070561105
# 0.006861261
# 0.006675923
# 0.006499458
