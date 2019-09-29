import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
# os.chdir("C:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class4\\use\\NG_sep")
os.chdir("C:\\Users\\youhe\\myfile\\CNN\\dataset\\arumihoiru\\split_anotate _argumentation\\1")
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
    if('jpg' in filename):
        with Image.open (filename) as im:
            print(filename)
            filename = filename.replace('.jpg','')
            im_rotate90 = im.transpose(Image.ROTATE_90)
            im_rotate90_m = im.transpose(Image.ROTATE_270)
            im_rotate90.save(filename +'_90.jpg')
            im_rotate90_m.save(filename +'_m_90.jpg')
            print(filename +'_90.jpg saved.')
            print(filename +'_90_m.jpg saved.')



'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
