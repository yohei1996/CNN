#ヒストグラム平滑化するためのファイル

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps

def hist_save(img2_hist,filename):
    hist , bins = np.histogram(img2_hist.ravel(),256,[0,256])
    # print(hist)
    plt.figure()
    plt.xlim(0,255)
    plt.plot(hist)
    print(hist)
    plt.xlabel("Pixel value" ,fontsize=20)
    plt.ylabel("Number of pixels" ,fontsize=20)
    plt.grid()
    plt.savefig(filename)

print(os.getcwd())
print(os.getcwd())

os.chdir("C:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\smple_image _flat")
print(os.listdir())
img_src = cv2.imread("mri1.png") 
img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
# 標準化
print("img1")
# print(img)

# cv2.imwrite('image2.png',img)
# hist_save(img,"normal2.png")

img_flat = cv2.equalizeHist(img)

cv2.imwrite('image_flat.png',img_flat)
# hist_save(img_flat,"flat2.png")





