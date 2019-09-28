# MINISTを読み込んでレイヤーAPIでCNNを構築するファイル
import tensorflow as tf
import numpy as np
import os

import tensorflow as tf
import glob
import numpy as np

import config as cf
from data_loader import DataLoader
from PIL import Image
from matplotlib import pylab as plt

dl = DataLoader(phase='Train', shuffle=True)
X_data , y_data = dl.shuffle_and_get()
# dl_test = DataLoader(phase='Test', shuffle=True)
X_data = np.reshape(X_data,[-1,cf.Height, cf.Width])


# plt.imshow(X_data[0])
# test_imgs, test_gts = dl_test.get_minibatch(shuffle=True)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"



# def load_img():
#     import cv2
#     img = cv2.imread("test.jpg").astype(np.float32)
#     img = cv2.resize(img, (cf.Width, cf.Height,1))
#     img = img[:,:,(2,1,0)]
#     img = img[np.newaxis, :]
#     img = img / 255.
#     return img

# with tf.Session(config=config) as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, "out.ckpt")

#     img = load_img()

#     pred = logits.eval(feed_dict={X: img, keep_prob: 1.0})[0]
#     pred_label = np.argmax(pred)
#     print(pred_label)

# X_data =  dataset['train_img']
# y_data = dataset['train_label']
# print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))
# X_test =dataset['test_img']
# y_test =dataset['test_label']
# print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))

# X_train, y_train = X_data[:50000,:], y_data[:50000]
# X_valid, y_valid = X_data[50000:,:], y_data[50000:]

# print('Training:   ', X_train.shape, y_train.shape)
# print('Validation: ', X_valid.shape, y_valid.shape)
# print('Test Set:   ', X_test.shape, y_test.shape)


