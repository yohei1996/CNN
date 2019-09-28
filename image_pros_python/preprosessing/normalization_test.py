# 正規化の数式を理解するためのファイル

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps

def hist_save(img3_hist,filename):
    hist , bins = np.histogram(img3_hist.ravel(),256,[0,256])
    print(hist)
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

os.chdir("c:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\smple_image")
img3_src = cv2.imread ("c:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\smple_image\\mri3.png") 
img3 = cv2.cvtColor(img3_src, cv2.COLOR_BGR2GRAY)
# 標準化
print("img3")
print(img3)

cv2.imwrite('image3.png',img3)
hist_save(img3,"normal.png")

cv2.imwrite('img3 - npmean(img3).png',img3 - np.mean(img3))
hist_save(img3 - np.mean(img3),'img3 - npmean(img3)_hist.png')

cv2.imwrite(' (img3 - np.mean(img3))np.std(img3).png', (img3 - np.mean(img3))/np.std(img3))
hist_save((img3 - np.mean(img3))/np.std(img3),'(img3 - np.mean(img3))np.std(img3)_hist.png')

img3_norm_10 = (img3 - np.mean(img3))/np.std(img3)*16+50
cv2.imwrite('img3_norm_16_50.png',img3_norm_10)
hist_save(img3_norm_10,"img3_norm_16_50_hist")

img3_norm_100 = (img3 - np.mean(img3))/np.std(img3)*16+100
cv2.imwrite('img3_norm_16_100.png',img3_norm_100)
hist_save(img3_norm_100,"img3_norm_16_100_hist")

img3_norm_101 = (img3 - np.mean(img3))/np.std(img3)*50+100
cv2.imwrite('img3_norm_50_100.png',img3_norm_101)
hist_save(img3_norm_101,"img3_norm_50_100_hist")
# print(img3_norm)
# cv2.imshow("Show BINARIZATION Image", img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow("Show BINARIZATION Image", img3_norm_10)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow("Show BINARIZATION Image", img3_norm_100)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
mean = np.mean(img3)
print("mean")
print(mean)
img3_minus_mean = img3-np.mean(img3)
print('img3_minus')
# print(img3_minus_mean)
std= np.std(img3)
print("std")
# print(std)
bunsuu = img3_minus_mean/std
print("bunsuu")
# print(bunsuu)
bunsuu_16 = bunsuu*16 
print("aa")

