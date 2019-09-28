import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from PIL import Image,ImageOps
print(os.getcwd())
os.chdir("\\Users\\nishitsuji\\Documents\\myfile\\CNN\\dataset\\DGIM_project\\Class1\\origin\\NG")
print(os.getcwd())



#im.save('savetest_'+filename)

filelist= os.listdir()
filelist.sort()
# print(filelist)
count=0

def filter2d(src, kernel, fill_value=-1):
                # カーネルサイズ
        m, n = kernel.shape
        
        # 畳み込み演算をしない領域の幅
        d = int((m-1)/2)
        h, w = src.shape[0], src.shape[1]
        
        # 出力画像用の配列
        if fill_value == -1: dst = src.copy()
        elif fill_value == 0: dst = np.zeros((h, w))
        else:
                dst = np.zeros((h, w))
                dst.fill(fill_value)

        # 畳み込み演算
        for y in range(d, h - d):
                for x in range(d, w - d):
                        dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)
        return dst


for filename in filelist:
        #if(count<2):#print(count)
        print(filename)
        count+=1
        if 'png' in filename:
                img_src = cv2.imread (filename) 
                img = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("Show BINARIZATION Image", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                filename = filename.replace('.png','')
                
                img = (img - np.mean(img))/np.std(img)-100
                cv2.imshow("Show BINARIZATION Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
               
                # #平均化フィルタ
                kernel = np.array([[1/9, 1/9, 1/9],
                                   [1/9, 1/9, 1/9],
                                   [1/9, 1/9, 1/9]])
                img = filter2d(img, kernel, -1) 
                cv2.imshow("Show BINARIZATION Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()      

   
                
                # 平滑化
                img = cv2.equalizeHist(img)
                cv2.imshow("Show BINARIZATION Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
              

                # 二値変換
                thresh = 100 #75がベスト
                max_pixel = 255
                ret, img = cv2.threshold(img,
                                        thresh,
                                        max_pixel,
                                        cv2.THRESH_BINARY)
                cv2.imshow("Show BINARIZATION Image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                


                #保存
                # cv2.imwrite("..//..//after_norm/NG//"+filename + "_norm.png",img)
                # 表示
                #cv2.imshow("Show BINARIZATION Image", img_dst)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
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
