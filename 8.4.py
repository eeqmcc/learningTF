# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

myimg = mpimg.imread('7068.jpg')
plt.imshow(myimg)
plt.axis('off')
plt.show()
print(myimg.shape)

full =  np.reshape(myimg, [1, myimg.shape[0], myimg.shape[1], myimg.shape[2]])
inputfull = tf.Variable(tf.constant(1.0, shape = [1, myimg.shape[0], myimg.shape[1], myimg.shape[2]]))

filter = tf.Variable(tf.constant([[-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0],
                                [-2.0, -2.0, -2.0], [0, 0, 0], [2.0, 2.0, 2.0],
                                [-1.0, -1.0, -1.0], [0, 0, 0], [1.0, 1.0, 1.0]], shape=[3, 3, 3, 1]))

op = tf.nn.conv2d(inputfull, filter, strides=[1, 1, 1, 1], padding='SAME')

o = tf.cast((op - tf.reduce_mean(op)) / (tf.reduce_max(op) - tf.reduce_min(op)) * 255, tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t, f = sess.run([o, filter], feed_dict = {inputfull: full})
    t = np.reshape(t, [myimg.shape[0], myimg.shape[1]])

    plt.imshow(t, cmap='Greys_r')
    plt.axis('off')
    plt.show()



