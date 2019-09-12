import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import theanoxla.nn as nn

w = T.Placeholder((3,), 'float32', name='w')
alpha = T.Placeholder((), 'float32', name='alpha')
t = T.Placeholder((), 'float32', name='t')
#step = T.Variable(1, name='var')
var=T.Variable(np.zeros(3), name='var')
q = t==2
update = T.cond(q, [t], 0, [2*t], 4)
#update = T.cond(step<4, step, lambda x: 0*x, step, lambda x:10*x)

#var, updates, step = nn.ExponentialMovingAverage(w, alpha)
#updates = {var:update, step:step+1}
#print(q.eval_value)
print('OUTOUT', T.get_output_for([update,q], {t:jax.numpy.array(4)}))
#print(q.eval_value)
#q.reset_value(True)
#print(q.eval_value)
print('OUTOUT', T.get_output_for([update,q], {t:jax.numpy.array(2)}))
exit()
train = theanoxla.function(w, alpha, t, outputs=[update])
#                           updates=updates)





import matplotlib.pyplot as plt
data = np.stack([np.ones(20), np.random.randn(20), np.zeros(20)], 1)
cost = list()
true_ema = [data[0]]
aa= 0.5
for j,i in enumerate(data):
    cost.append(train(i, aa, j))
    true_ema.append(aa*true_ema[-1]+(1-aa)*i)
    print(cost[-1])
cost = np.asarray(cost)
true = np.asarray(true_ema)[1:]
print(np.mean(np.isclose(cost, true)))

plt.subplot(131)
plt.plot(data[:, 0])
plt.plot(cost[:, 0])
plt.plot(true[:, 0])

plt.subplot(132)
plt.plot(data[:, 1])
plt.plot(cost[:, 1])
plt.plot(true[:, 1])

plt.subplot(133)
plt.plot(data[:, 2])
plt.plot(cost[:, 2])
plt.plot(true[:, 2])

plt.show()
