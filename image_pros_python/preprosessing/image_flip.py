import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
# os.chdir("C:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class4\\use\\NG_sep")
os.chdir("C:\\Users\\youhe\\myfile\\CNN\\dataset\\arumihoiru\\split_anotate _argumentation\\1")
print(os.getcwd())

filelist= os.listdir()
filelist.sort()
print(filelist)
count=0
for filename in filelist:
    #print(count)
    #print(filename)
    count+=1
    if('jpg' in filename):
        img = cv2.imread(filename)
        filename = filename.replace('.jpg','')

        img_x = cv2.flip(img, 0)
        img_y = cv2.flip(img, 1)
        img_xy = cv2.flip(img, -1)
        
        cv2.imwrite( filename + '_x.jpg', img_x)
        cv2.imwrite( filename + '_y.jpg', img_y)
        cv2.imwrite( filename + '_xy.jpg', img_xy)
        print(filename +"saved!")


'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
