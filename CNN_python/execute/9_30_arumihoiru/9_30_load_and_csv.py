import numpy as np
import datetime as dt
import os
from matplotlib import pylab as plt
import matplotlib.ticker as ticker
import csv
os.chdir("c:\\Users\\youhe\\myfile\\CNN\\CNN_python\\execute\\9_30_arumihoiru")

accuracy_load = np.load(file="./save/9_30_data_save/2019_09_30accuracy.npy")
loss_load = np.load(file="./save/9_30_data_save/2019_09_30loss.npy")
arr =[]
for xv,yv in zip(accuracy_load,loss_load):
    arr.append([xv,yv])
with open("./save/condition_and_result/acc_and_loss.csv","w",encoding="Shift_jis") as f:
    writer = csv.writer(f,lineterminator="\n")
    writer.writerows(arr)