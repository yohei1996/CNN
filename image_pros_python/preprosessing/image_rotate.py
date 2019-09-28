import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
# os.chdir("C:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class4\\use\\NG_sep")
os.chdir("\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class1\\after_norm\\NG")
print(os.getcwd())

'''
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
        with Image.open (filename) as im:
            print(filename)
            filename = filename.replace('.png','')
            im_rotate90 = im.transpose(Image.ROTATE_90)
            im_rotate90_m = im.transpose(Image.ROTATE_270)
            im_rotate90.save(filename +'_90.png')
            im_rotate90_m.save(filename +'_m_90.png')
            print(filename +'_90.png saved.')
            print(filename +'_90_m.png saved.')



'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
