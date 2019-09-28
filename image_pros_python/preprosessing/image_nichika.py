import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
os.chdir("C:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class4\\nichika\\NG")
print(os.getcwd())



#im.save('savetest_'+filename)

filelist= os.listdir()
filelist.sort()
# print(filelist)
count=0
for filename in filelist:
    #print(count)
    print(filename)
    count+=1
    if 'png' in filename:
        img_src = cv2.imread (filename) 
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        filename = filename.replace('.png','')
        # 二値変換
        thresh = 145
        max_pixel = 255
        ret, img_dst = cv2.threshold(img_gray,
                                    thresh,
                                    max_pixel,
                                    cv2.THRESH_BINARY)
        #保存
        cv2.imwrite("..//..//after_nitika//NG//"+filename + "_nitika.png",img_dst)
        # 表示
        # cv2.imshow("Show BINARIZATION Image", img_dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(filename)
        print(filename +'_flp.png saved.')
        # im_mirror = ImageOps.mirror(im)
        # im_mirror.save(filename +'_mir.png')
        # print(filename +'_mir.png saved.')            



'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
