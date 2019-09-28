import numpy as np
import datetime as dt
import os
from matplotlib import pylab as plt
import matplotlib.ticker as ticker
os.chdir("c:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\9_12_DGIM_validation\\")

folder_name = "9_12_data_save"
if folder_name not in  os.listdir():
    os.chdir("./"+folder_name)
os.chdir("./"+folder_name)

loss_load = np.load(file="./"+"2019_09_14loss.npy")
# accuracy_load = np.load(file="./"+"2019_09_14loss.npy")
print("loss_load:",loss_load)

x = np.linspace(0,20,20)
y = loss_load
fig = plt.figure(figsize=(12,8))


ax = fig.add_subplot(111)
# ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,label="loss")
# plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.ylabel("Loss")
plt.show()