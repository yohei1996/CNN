import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps



#im.save('savetest_'+filename)
print(os.getcwd())

print(os.getcwd())

os.chdir("c:\\Users\\youhe\\myfile\\CNN\\dataset\\arumihoiru\\original\\")
filelist= os.listdir()
# os.chdir("..\\")
# if "split" not in os.listdir():
#     os.mkdir("split")
# os.chdir("..\\split\\")
save_path = "..\\split\\"
filelist.sort()
print(filelist)
count=0
img_num = 1
for filename in filelist:
    print(filename)
    count+=1
    if 'jpg' in filename:
        img_src = cv2.imread(filename)
        img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        img = cv2.resize(img,dsize=(1536,2048))
        print(img.shape)
        filename = filename.replace('.jpg','')
        
        # cv2.imshow("Show BINARIZATION Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # img = cv2.imread('sample.jpg')

        num_vsplits = 8  # 垂直方向の分割数
        num_hsplits = 6  # 水平方向の分割数

        # 均等に分割できないと np.spllt() が使えないので、
        # 除算したときに余りがでないように画像の端数を切り捨てる。
        h, w = img.shape[:2]
        crop_img = img[:h // num_vsplits * num_vsplits, :w // num_hsplits * num_hsplits]
        print('{} -> {}'.format(img.shape, crop_img.shape))  # (480, 640, 3) -> (480, 637, 3)   
      
        # 分割する。
        out_imgs = []
        for h_img in np.vsplit(crop_img, num_vsplits):  # 垂直方向に分割する。
            for v_img in np.hsplit(h_img, num_hsplits):  # 水平方向に分割する。
                out_imgs.append(v_img)
        out_imgs = np.array(out_imgs)
        print(out_imgs.shape)  
        for image in out_imgs:
            file_path = save_path + str(img_num) + ".jpg"
            cv2.imwrite(file_path,image)
            print(file_path+" saved!")
            img_num+=1






'''
blur3 = cv2.blur(img,(3,3))
blur5 = cv2.blur(img,(5,5))
edges = cv2.Canny(img,100,200)
'''
