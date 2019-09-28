import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
os.chdir("\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class1\\origin\\NG")
print(os.getcwd())


# 平滑化
img = cv2.equalizeHist(img)
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
