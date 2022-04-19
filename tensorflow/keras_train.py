
from matplotlib.pyplot import savefig
import tensorflow as tf
import torch
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

import tensorboardX
from classify import *
from datas import FrameDatas
import cv2
import json
import random
# from keras.utils import multi_gpu_model


def calculate_acc(preds, y):
    p1 = np.argmax(preds, axis=1)
    p2 = np.argmax(y, axis=1)
    correct = np.sum((p1 == p2).astype(np.int))
    acc = correct / len(p1)
    return correct, len(p1), acc


def train(save_path, sess, batch_size, lr, epochs, train_data, test_data):
    
    with tf.Graph().as_default(), tf.compat.v1.Session(config=sess) as sess:
        inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, 512,384, 6), name='inputs')
        labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size, 4), name='labels')

        model = OpticalFlowMotion()
        pred = model(inputs)
        entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels, pred)
        loss = tf.reduce_mean(entropy_loss)

        global_step = tf.Variable(0, trainable=False)
        add_global = global_step.assign_add(1)
        best_acc = 0.0
        train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(entropy_loss)
        sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
        for epoch in range(epochs):
            for step, data in enumerate(tqdm(train_data), 1):
                x, y = data
                # print('x:', x.shape)
                _, _, losses, preds = sess.run([train_opt, add_global, loss, pred], feed_dict={inputs: x, labels:y})
                _, _, acc = calculate_acc(preds, y)
                if step % 10 == 0:
                    print('epoch: [%d]/[%d], steps: [%d]/[%d], loss = %.6f, acc = %.6f' %(epoch+1, epochs, step, len(train_data), losses, acc)) 

            corrects = 0
            total = 0
            for step, data in enumerate(tqdm(test_data), 1):
                x, y = data
                preds = sess.run([pred], feed_dict={inputs: x, labels:y})
                correct, cnt, _ = calculate_acc(preds, y)
                corrects += correct
                total += cnt
            
            acc = corrects / total
            saver = tf.train.Saver()
            print('***** epoch %d/%d done, acc = %.6f *********'%(epoch+1, epochs, acc)) 
            if acc > best_acc:
                best_acc = acc
                print('... saving ...')
                saver.save(sess, save_path+'/cls-' +str(epoch+1)+'_.ckpt', write_meta_graph =True)
                                  

if __name__ =='__main__':
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
  
    print('==> start training loop ...')

    data_file = '../MotionClassify/datas/sample.json'
    save_folder = os.path.join('outputs', 'keras')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    samples = json.load(open(data_file, 'r'))
    random.shuffle(samples)

    slice = int(len(samples) * 0.9)
    train_sample = samples[:slice]
    test_sample = samples[slice:]

    with open(os.path.join(save_folder, 'test.json'), 'w') as f:
        json.dump(test_sample, f)
    
    batch_size = 8
    train_data = FrameDatas(train_sample, batch_size, (384, 512))   
    test_data = FrameDatas(test_sample, batch_size, (384, 512))

    '''
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, batch_size=8, epochs=1, steps_per_epoch=len(train_data))
    model.save(save_folder)
    test_loss, test_acc = model.evaluate(test_data)
    print('Test acc: ', test_acc)
    '''

    train(save_folder, sess_config,  batch_size, 0.005, 10, train_data, test_data)
