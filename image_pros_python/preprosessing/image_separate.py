import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
os.chdir("C:\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class4\\OK")


from PIL import Image
import sys

from datetime import datetime

def ImgSplit(im):
    # 読み込んだ画像を200*200のサイズで54枚に分割する
    height = 256
    width = 256

    # 縦の分割枚数
    for h1 in range(2):
        # 横の分割枚数
        for w1 in range(2):
            w2 = w1 * height
            h2 = h1 * width
            yield im.crop((w2, h2, width + w2, height + h2))





# 画像の読み込み
im = Image.open('1.png')


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
            # im_flip = ImageOps.flip(im)
            # im_flip.save(filename +'_flp.png')
            # print(filename +'_flp.png saved.')
            for position , ig in zip(['UL','UR','LL','LR'],ImgSplit(im)):
                # 保存先フォルダの指定
                #ig.show()
                ig.save('..\\OK_sep\\'+filename + position + '.png')
            print(filename + position + '.png saved.')            



'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
