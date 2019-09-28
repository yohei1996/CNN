import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
os.chdir("\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class1\\origin\\NG")
print(os.getcwd())

def hist_save(img2_hist,filename):
    hist , bins = np.histogram(img2_hist.ravel(),256,[0,256])
    print(hist)
    plt.figure()
    plt.xlim(0,255)
    plt.plot(hist)
    print(hist)
    plt.xlabel("Pixel value" ,fontsize=20)
    plt.ylabel("Number of pixels" ,fontsize=20)
    plt.grid()
    plt.savefig(filename)

img_src = cv2.imread ("1.png") 
img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
cv2.imshow("Show BINARIZATION Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 平滑化
img = cv2.equalizeHist(img)
cv2.imshow("Show BINARIZATION Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# #平均化フィルタ
# kernel = np.array([[1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9],
#                    [1/9, 1/9, 1/9]])
# 方法1(NumPyで実装)

img = filter2d(img, kernel, -1)


# 標準化
img = (img_gray - np.mean(img_gray))/np.std(img_gray)*16+100

cv2.imshow("Show BINARIZATION Image", img)
# 二値変換
thresh = 0 #75がベスト
max_pixel = 255
ret, img = cv2.threshold(img,
                        thresh,
                        max_pixel,
                        cv2.THRESH_BINARY)
cv2.imshow("Show BINARIZATION Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




                # #保存
                # cv2.imwrite("..//..//after_norm/OK//"+filename + "_norm.png",img)
                # # 表示
                # #cv2.imshow("Show BINARIZATION Image", img_dst)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # print(filename)
                # print(filename +'_flp.png saved.')
                # # im_mirror = ImageOps.mirror(im)
                # im_mirror.save(filename +'_mir.png')
                # print(filename +'_mir.png saved.')            



'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
