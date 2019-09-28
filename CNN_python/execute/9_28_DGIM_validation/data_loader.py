import os
import glob
import cv2
import numpy as np

import config as cf
import tensorflow as tf
import random
from PIL import Image
from matplotlib import pylab as plt

class DataLoader():

    def __init__(self,phase='Train', shuffle=False):
        self.datas = []
        self.imgs = []
        # self.imgs = np.array([])
        self.labels = []
        self.last_mb = 0
        self.phase = phase
        self.gt_count = [0 for _ in range(cf.Class_num)]
        self.prepare_datas(shuffle=shuffle)

    def prepare_datas(self, shuffle=True):
        folder_path = "C://Users//nishitsuji//Documents//myfile//python_tensorflow//dataset"
        print('------------')
        label_paths = glob.glob(folder_path + '//*')

        load_count = 0

        for label_path in label_paths:
            label = 0 if "OK" in label_path else 1
            files = glob.glob(label_path+'//*')
            for img_path in files:
                gt = self.get_gt(img_path)
                img = self.load_image(img_path)
                img = np.reshape(img,[cf.Height, cf.Width, 1])
                # self.imgs = np.append(self.imgs,img)
                self.imgs.append(img)
                self.labels.append(label)
                load_count +=1
                if load_count %100 ==0:
                    print(load_count) 
            print(' - {} - {} datas -> loaded {}'.format(label_path, len(files), load_count))


    def shuffle_and_get(self):
        p = list(zip(self.imgs,self.labels))
        random.shuffle(p)
        # for img,label in self.datas:
        #     self.imgs =np.append(imgs,img)
        #     self.labels =np.append(labels,label)
        imgs,labels = zip(*p)
        return imgs,labels
    

    def display_data_total(self):

        print('   Total data: {}'.format(len(self.datas)))


    def display_gt_statistic(self):

        print()
        print('  -*- Training label statistic -*-')
        self.display_data_total()

        for i, gt in enumerate(self.gt_count):
            print('   - {} : {}'.format(cf.Class_label[i], gt))



    def set_index(self, shuffle=True):
        self.data_n = len(self.datas)

        self.indices = np.arange(self.data_n)

        if shuffle:
            np.random.seed(0)
            np.random.shuffle(self.indices)







    def get_gt(self, img_name):

        for ind, cls in enumerate(cf.Class_label):
            if cls in img_name:
                return ind

        raise Exception("Class label Error {}".format(img_name))    


    ## Below functions are for data augmentation

    def load_image(self, img_name, h_flip=False, v_flip=False):

        ## Image load

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img is None:
            raise Exception('file not found: {}'.format(img_name))

        img = cv2.resize(img, (cf.Width, cf.Height))
        # img = img[:, :, (2,1,0)]
        img = img / 255. #☆これどうしよう..?

        ## Horizontal flip
        if h_flip:
            img = img[:, ::-1, :]

        ## Vertical flip
        if v_flip:
            img = img[::-1, :, :]

        return img



    def data_augmentation(self, h_flip=False, v_flip=False):

        print()
        print('   ||   -*- Data Augmentation -*-')
        if h_flip:
            self.add_horizontal_flip()
            print('   ||    - Added horizontal flip')
        if v_flip:
            self.add_vertical_flip()
            print('   ||    - Added vertival flip')
        print('  \  /')
        print('   \/')



    def add_horizontal_flip(self):

        ## Add Horizontal flipped image data

        new_data = []

        for data in self.datas:
            _data = {'img_path': data['img_path'],
                     'gt_path': data['gt_path'],
                     'h_flip': True,
                     'v_flip': data['v_flip']
            }

            new_data.append(_data)

            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1

        self.datas.extend(new_data)



    def add_vertical_flip(self):

        ## Add Horizontal flipped image data

        new_data = []

        for data in self.datas:
            _data = {'img_path': data['img_path'],
                     'gt_path': data['gt_path'],
                     'h_flip': data['h_flip'],
                     'v_flip': True
            }

            new_data.append(_data)

            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1

        self.datas.extend(new_data)