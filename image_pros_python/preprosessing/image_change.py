import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
os.chdir("\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class1\\after_norm\\NG")
print(os.getcwd())

'''
filename='1.png'
im=Image.open(filename)
im.show()
im.show()
im = ImageOps.mirror(im)
im.show()
#im.save('savetest_'+filename)
'''

filelist= os.listdir()
filelist.sort()
print(filelist)
count=0
for filename in filelist:
    #print(count)
    #print(filename)
    count+=1
    if('png' in filename):
    # if('png' in filename)and(count<5):
        with Image.open (filename) as im:
            print(filename)
            filename = filename.replace('.png','')
            im_flip = ImageOps.flip(im)
            im_flip.save(filename +'_flp.png')
            print(filename +'_flp.png saved.')
            im_mirror = ImageOps.mirror(im)
            im_mirror.save(filename +'_mir.png')
            print(filename +'_mir.png saved.')            



'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
