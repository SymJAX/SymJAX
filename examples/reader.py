import pickle
import numpy as np
import matplotlib.pyplot as plt


file1 = open('saveit_0.pkl','rb')
file2 = open('saveit_4.pkl','rb')
file3 = open('saveit_8.pkl','rb')

data1 = pickle.load(file1)
data2 = pickle.load(file2)
data3 = pickle.load(file3)

file1.close()
file2.close()
file3.close()

DATA1 = np.array(data1[1::3]) * 100
DATA2 = np.array(data2[1::4]) * 100
DATA3 = np.array(data3[1::4]) * 100

TDATA1 = np.array(data1[0::3])
TDATA2 = np.array(data2[0::4])
TDATA3 = np.array(data3[0::4])

FDATA1 = np.array(data1[2::3])
FDATA2 = np.array(data2[2::4])
FDATA3 = np.array(data3[2::4])


plt.figure(figsize=(6,4))

plt.subplot(121)
plt.title('Accu')
plt.plot(TDATA1[:,1], 'b')
plt.plot(TDATA2[:,1], 'g')
plt.plot(TDATA3[:,1], 'r')

plt.subplot(122)
plt.title('Loss (CE)')
plt.plot(TDATA1[:,0], 'b')
plt.plot(TDATA2[:,0], 'g')
plt.plot(TDATA3[:,0], 'r')
plt.savefig('accu_loss.png')
plt.close()

plt.figure(figsize=(6,4))
plt.subplot(121)
plt.title('AUC')
plt.plot(DATA1[:,1], 'b')
plt.plot(DATA2[:,1], 'g')
plt.plot(DATA3[:,1], 'r')

plt.subplot(122)
plt.title('Accuracy')
plt.plot(DATA1[:,0], 'b')
plt.plot(DATA2[:,0], 'g')
plt.plot(DATA3[:,0], 'r')
plt.savefig('auc_accu.png')
plt.close()

plt.figure()
plt.subplot(121)
plt.imshow(data2[2][0,0], aspect='auto')
plt.subplot(122)
plt.imshow(data2[-1][0,0], aspect='auto')
plt.savefig('filter.png')
plt.close()

plt.figure()
plt.subplot(131)
plt.imshow(FDATA1[-1, 0], aspect='auto')
plt.subplot(132)
plt.imshow(FDATA2[-1, 0], aspect='auto')
plt.subplot(133)
plt.imshow(FDATA3[-1, 0], aspect='auto')
plt.savefig('repr.png')
plt.close()

