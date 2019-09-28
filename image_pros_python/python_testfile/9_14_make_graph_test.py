from matplotlib import pylab as plt
import numpy as np

x = np.linspace(0,10,10)
loss = range(10)

fig = plt.figure(figsize=(12,8))


ax = fig.add_subplot(111)
# ax = fig.add_axes([0,0,1,1])
ax.plot(x,loss,label="loss")
# plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
