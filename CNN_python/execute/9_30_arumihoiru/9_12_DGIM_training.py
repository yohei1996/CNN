# MINISTを読み込んでレイヤーAPIでCNNを構築するファイル
import tensorflow as tf
import numpy as np
import os
import struct
import pickle
import glob
import config as cf
from data_loader import DataLoader
from PIL import Image
from matplotlib import pylab as plt
import random
import cv2
import datetime as dt

class ConvNN(object):
    def __init__(self, batchsize=cf.batch_size,
                 epochs=20, learning_rate=1e-4, 
                 dropout_rate=0.5,
                 shuffle=True, random_seed=None):
        np.random.seed(random_seed)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        self.loss_data = []
        self.accuracy_data = []
        
        print("グラフ作成")
        g = tf.Graph()
        with g.as_default():
            ## set random-seed:
            tf.set_random_seed(random_seed)
            
            ## build the network:
            self.build()

            ## initializer
            self.init_op = tf.global_variables_initializer()

            ## saver
            self.saver = tf.train.Saver()
            
        ## create a session
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.visible_device_list="0"

        self.sess = tf.Session(graph=g,config=config)
                
    def build(self):
        print("build")
        ## Placeholders for X and y:
        tf_x = tf.placeholder(tf.float32, 
                              shape=[None, cf.Height*cf.Width],
                              name='tf_x')
        tf_y = tf.placeholder(tf.int32, 
                              shape=[None],
                              name='tf_y')
        is_train = tf.placeholder(tf.bool, 
                              shape=(),
                              name='is_train')

        ## reshape x to a 4D tensor: 
        ##  [batchsize, width, height, 1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, cf.Height,cf.Width, 1],
                              name='input_x_2dimages')
        ## One-hot encoding:
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=2,
                              dtype=tf.float32,
                              name='input_y_onehot')

        ## 1st layer: Conv_1
        h1 = tf.layers.conv2d(tf_x_image, 
                              kernel_size=(5, 5), 
                              filters=32, 
                              activation=tf.nn.relu)
        ## MaxPooling
        h1_pool = tf.layers.max_pooling2d(h1, 
                              pool_size=(2, 2), 
                              strides=(2, 2))
        ## 2n layer: Conv_2
        h2 = tf.layers.conv2d(h1_pool, kernel_size=(5,5), 
                              filters=64, 
                              activation=tf.nn.relu)
        ## MaxPooling 
        h2_pool = tf.layers.max_pooling2d(h2, 
                              pool_size=(2, 2), 
                              strides=(2, 2))

        ## 3rd layer: Fully Connected
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, 
                              shape=[-1, n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 1024, 
                              activation=tf.nn.relu)
        print(h3)

        ## Dropout
        h3_drop = tf.layers.dropout(h3, 
                              rate=self.dropout_rate,
                              training=is_train)
        
        ## 4th layer: Fully Connected (linear activation)
        h4 = tf.layers.dense(h3_drop, 2, 
                              activation=None)
        print(h4)
        ## Prediction
        predictions = {
            'probabilities': tf.nn.softmax(h4, 
                              name='probabilities'),
            'labels': tf.cast(tf.argmax(h4, axis=1), 
                              tf.int32, name='labels')}
        
        ## Loss Function and Optimization
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=h4, labels=tf_y_onehot),
            name='cross_entropy_loss')
        
        ## Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss,
                                       name='train_op')

        ## Finding accuracy
        correct_predictions = tf.equal(
            predictions['labels'], 
            tf_y, name='correct_preds')
        
        accuracy = tf.reduce_mean(
            tf.cast(correct_predictions, tf.float32),
            name='accuracy')

    def save(self, epoch, path='./tflayers-model/'):
        dt_now = dt.datetime.now()
        file_name_date=dt_now.strftime('%Y_%m_%d')
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Saving model in %s' % path)
        self.saver.save(self.sess, 
                        os.path.join(path,'model.ckpt'),
                        global_step=epoch)
        
    def load(self, epoch, path):
        print('Loading model from %s' % path)
        self.saver.restore(self.sess, 
             os.path.join(path, 'model.ckpt-%d' % epoch))
        
    def train(self, training_set, 
              validation_set=None,
              initialize=True,
              return_proba = False):
        ## initialize variables
        print("start_init")
        if initialize:
            self.sess.run(self.init_op)

        self.train_cost_ = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])

        for epoch in range(1, self.epochs + 1):
            batch_gen = batch_generator(X_data, y_data, batch_size=cf.batch_size,
                                 shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x,batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x, 
                        'tf_y:0': batch_y,
                        'is_train:0': True} ## for dropout
                loss, _ = self.sess.run(
                        ['cross_entropy_loss:0', 'train_op'], 
                        feed_dict=feed)
                avg_loss += loss
                
            print('Epoch %02d: Training Avg. Loss: '
                  '%7.3f' % (epoch, avg_loss), end=' ')
            self.loss_data.append(avg_loss)

            feed = {'tf_x:0': validation_set[0],
                'is_train:0': False} ## for dropout
            if return_proba:
                preds = self.sess.run('probabilities:0',
                                     feed_dict=feed)
            else:
                preds = self.sess.run('labels:0',
                                     feed_dict=feed)

            print('Test Accuracy: %.2f%%' % (100*np.sum(validation_set[1] == preds)/len(validation_set[1])))
            # if validation_set is not None:
            #     feed = {'tf_x:0': batch_x, 
            #             'tf_y:0': batch_y,
            #             'is_train:0': False} ## for dropout
            #     valid_acc = self.sess.run('accuracy:0',
            #                               feed_dict=feed)
            #     print('Validation Acc: %7.3f' % valid_acc)
            #     self.accuracy_data.append(valid_acc)
            # else:
            #     print()
                    
    def predict(self, X_test, return_proba = False):
        feed = {'tf_x:0': X_test,
                'is_train:0': False} ## for dropout
        if return_proba:
            return self.sess.run('probabilities:0',
                                 feed_dict=feed)
        else:
            return self.sess.run('labels:0',
                                 feed_dict=feed)

    def loss_accuracy_save(self):
        current_path = os.getcwd()
        os.chdir("c:\\Users\\youhe\\myfile\\CNN\\CNN_python\\execute\\9_28_DGIM_validation\\save")

        folder_name = "9_12_data_save"
        if folder_name not in  os.listdir():
            os.mkdir("./"+folder_name)
        os.chdir("./"+folder_name)

        dt_now = dt.datetime.now()
        file_name_date=dt_now.strftime('%Y_%m_%d')

        loss = self.loss_data
        accuracy = self.accuracy_data
        print("loss:",loss)
        print("accuracy:",accuracy)
        np.save("./"+file_name_date+"loss",loss)
        np.save("./"+file_name_date+"accuracy",accuracy)
        os.chdir(current_path)




def batch_generator(X, y, batch_size=50, 
                    shuffle=False, random_seed=None):
    
    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

def show_image(X_data,y_data):
    ok_count=0
    ng_count=0
    num_arrey_ok =[]
    num_arrey_ng =[]
    while ok_count != 2 or ng_count != 2:
        num = random.choice(range(len(X_data)))
        if y_data[num]==0 and ok_count!=2:
            ok_count +=1
            num_arrey_ok.append(num)
        if y_data[num]==1 and  ng_count!=2:
            ng_count +=1
            num_arrey_ng.append(num)
    # fig, ax = plt.subplots()
    imgs=[]
    imgs.append(np.reshape(X_data[num_arrey_ng[0]],[cf.Height, cf.Width]))
    imgs.append(np.reshape(X_data[num_arrey_ng[1]],[cf.Height, cf.Width]))
    imgs.append(np.reshape(X_data[num_arrey_ok[0]],[cf.Height, cf.Width]))
    imgs.append(np.reshape(X_data[num_arrey_ok[1]],[cf.Height, cf.Width]))
    fig, ax_list = plt.subplots(2,2, figsize=(5,5))
    for sub_img, ax in zip(imgs, ax_list.ravel()):
        ax.imshow(sub_img[..., ::-1],'gray')
        ax.set_axis_off()
    plt.show()

dl = DataLoader(phase='Train', shuffle=True)
X_data , y_data = dl.shuffle_and_get()
# dl_test = DataLoader(phase='Test', shuffle=True)
# print(X_data.shape)
X_data = np.reshape(X_data,[-1,cf.Height*cf.Width])
# print(X_data.shape)


# show_image(X_data,y_data)


    # plt.imshow(X_data[0])
    # test_imgs, test_gts = dl_test.get_minibatch(shuffle=True)
print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))

X_train, y_train = X_data[:-300], y_data[:-300]
X_valid, y_valid = X_data[-300:], y_data[-300:]

X_test,y_test = zip(*random.sample(list(zip(X_data,y_data)),100))
X_test = np.array(X_test)
# print('Training:   ',X_train.shape, y_train.shape)
# print('Validation: ',X_valid.shape, y_valid.shape)
# print('Test Set:   ', X_test.shape, y_test.shape)




mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals)/std_val
X_valid_centered = (X_valid - mean_vals)/std_val
X_test_centered = (X_test - mean_vals)/std_val

del X_data, y_data, X_train, X_valid , X_test

print("loadingCNN")
cnn = ConvNN(random_seed=123)


print("trainCNN")
cnn.train(training_set=(X_train_centered, y_train), 
          validation_set=(X_valid_centered, y_valid))

print("saveCNN")
cnn.save(epoch=20,path='./9_16_tflayers-model/')

print("LOSS_AND_CUURACY_SAVE")
cnn.loss_accuracy_save()

del cnn

print("loadCNN")
cnn2 = ConvNN(random_seed=123)

cnn2.load(epoch=20,path='./9_16_tflayers-model/')

# print(cnn2.predict(X_test_centered[:10,:]))

preds = cnn2.predict(X_test_centered)

print('Test Accuracy: %.2f%%' % (100*
      np.sum(y_test == preds)/len(y_test)))

# elapsed_time = time.time() - start
start = time.time()
import time
print("テスト時間")
print("ファイル枚数："+str(len(X_test_centered)))
print("サイズ："+str(X_test_centered[0].shape))
print ("1000枚あたりの時間:{0}".format(elapsed_time) + "[sec]")
print ("1枚あたりの時間:{0}".format(elapsed_time/len(X_test_centered)) + "[sec]")
print("STOP!")