import numpy as np
import matplotlib.pyplot as plt

labels = np.array([0,100, 300, 600, 1000, 3000])
acc_sup = np.array([99.38, 99.38,99.38,99.38,99.38,99.38])
acc = np.array([91.16, 93.51, 96.86, 98.03, 98.85])
acc_pl = np.array([90.56, 96.68, 97.84, 98.10, 99.10])

plt.plot(labels,acc_sup,'--')
#plt.plot(labels[1:],acc,'-o')
plt.plot(labels[1:],acc_pl,'-o')
plt.xlabel('Size of labeled data')
plt.ylabel('Accuracy')
plt.legend(['Entire labeled set','Pseudo-labeling'])
plt.xlim([0,3050])
plt.show()


