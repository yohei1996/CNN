import numpy as np
import datetime as dt
import os
from matplotlib import pylab as plt
os.chdir("c:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\9_12_DGIM_validation\\")

os.makedirs
folder_name = "9_12_data_save"
if folder_name not in  os.listdir():
    os.chdir("./"+folder_name)
os.chdir("./"+folder_name)

dt_now = dt.datetime.now()
file_name_date=dt_now.strftime('%Y_%m_%d')

loss = np.random.randn(20)*10
print("loss:",loss)
np.save("./"+file_name_date+"loss",loss)


loss_load = np.load(file="./"+file_name_date+"loss"+".npy")
print("loss_load:",loss_load)

x = np.linspace(0,10,20)
y = loss_load
fig = plt.figure(figsize=(12,8))


ax = fig.add_subplot(111)
# ax = fig.add_axes([0,0,1,1])
ax.plot(x,y,label="loss")
# plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()