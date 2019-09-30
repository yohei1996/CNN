import numpy as np
import datetime as dt
import os
from matplotlib import pylab as plt
import matplotlib.ticker as ticker

os.chdir("c:\\Users\\youhe\\myfile\\CNN\\CNN_python\\execute\\9_30_arumihoiru\\save\\9_30_data_save\\")

accuracy_load = np.load(file="./9_30_data_save/2019_09_30accuracy.npy")
loss_load = np.load(file="./9_30_data_save/2019_09_30loss.npy")
print("accuracy_load:",accuracy_load)

x = np.linspace(0,x.shape[0],x.shape[0])
y = loss_load
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
# ax = fig.add_axes([0,0,1,1])
ax.plot(x,y*100,label="Accuracy",color="red")
# plt.legend()
# plt.title("Accuracy")
plt.xlabel("Epoch")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylabel("accuracy")
plt.show()