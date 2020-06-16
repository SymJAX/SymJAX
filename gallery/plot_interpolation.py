import matplotlib.pyplot as plt

import symjax
import symjax.tensor as T

w = T.Placeholder((3,), 'float32', name='w')
w_interp1 = T.interpolation.upsample_1d(w, repeat=4, mode='nearest')
w_interp2 = T.interpolation.upsample_1d(w, repeat=4, mode='linear',
                                        boundary_condition='mirror')
w_interp3 = T.interpolation.upsample_1d(w, repeat=4, mode='linear',
                                        boundary_condition='periodic')
w_interp4 = T.interpolation.upsample_1d(w, repeat=4)

f = symjax.function(w, outputs=[w_interp1, w_interp2, w_interp3, w_interp4])

samples = f([1, 2, 3])
fig = plt.figure(figsize=(12,4))
plt.subplot(411)
plt.plot(samples[0], 'xg',linewidth=3, markersize=15)
plt.plot([0, 5, 10], [1, 2, 3], 'ok', alpha=0.5)
plt.title('nearest-periodic')
plt.xticks([])

plt.subplot(412)
plt.plot(samples[1], 'xg', linewidth=3, markersize=15)
plt.plot([0, 5, 10], [1, 2, 3], 'ok', alpha=0.5)
plt.title('linear-mirror')
plt.xticks([])

plt.subplot(413)
plt.plot(samples[2], 'xg', linewidth=3, markersize=15)
plt.plot([0, 5, 10], [1, 2, 3], 'ok', alpha=0.5)
plt.title('linear-periodic')
plt.xticks([])

plt.subplot(414)
plt.plot(samples[3], 'xg',linewidth=3, markersize=15)
plt.plot([0, 5, 10], [1, 2, 3], 'ok', alpha=0.5)
plt.title('constant-0')


plt.suptitle('1D linear interpolation')

plt.tight_layout()

