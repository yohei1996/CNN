import numpy as np
import datetime as dt
import os
from matplotlib import pylab as plt
import matplotlib.ticker as ticker
import csv
os.chdir("c:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\9_12_DGIM_validation\\")

folder_name = "9_12_data_save"
if folder_name not in  os.listdir():
    os.chdir("./"+folder_name)
os.chdir("./"+folder_name)

accuracy_load = np.load(file="./"+"2019_09_14accuracy.npy")
load_load = np.load(file="./"+"2019_09_14loss.npy")
# accuracy_load = np.load(file="./"+"2019_09_14accuracy.npy")
# print("accuracy_load:",accuracy_load)

# x = np.linspace(0,20,20)
# y = accuracy_load
arr =[]
for xv,yv in zip(accuracy_load,load_load):
    arr.append([xv,yv])
with open("../../result_images/acc_and_loss.csv","w",encoding="Shift_jis") as f:
    writer = csv.writer(f,lineterminator="\n")
    writer.writerows(arr)