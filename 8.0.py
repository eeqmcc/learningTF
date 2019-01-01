# -*- coding:utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow as tf

# shape: [batch, in_height, in_width, in_channel]
input1 = tf.Variable(tf.constant(1.0, shape=[1,5,5,1]))
input2 = tf.Variable(tf.constant(1.0, shape=[1,5,5,2]))
input3 = tf.Variable(tf.constant(1.0, shape=[1,4,4,1]))

# filter shape: [filter_height, filter_width, in_channels, out_channels]
filter1 = tf.Variable(tf.constant([-1.0, 0, 0, -1], shape=[2, 2, 1, 1]))
filter2 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 1, 2]))
filter3 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 1, 3]))
filter4 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1, -1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 2, 2]))
filter5 = tf.Variable(tf.constant([-1.0, 0, 0, -1, -1.0, 0, 0, -1], shape=[2, 2, 2, 1]))


op1 = tf.nn.conv2d(input1, filter1, strides=[1, 2, 2, 1], padding='SAME')
op2 = tf.nn.conv2d(input1, filter2, strides=[1, 2, 2, 1], padding='SAME')
op3 = tf.nn.conv2d(input1, filter3, strides=[1, 2, 2, 1], padding='SAME')
op4 = tf.nn.conv2d(input2, filter4, strides=[1, 2, 2, 1], padding='SAME')
op5 = tf.nn.conv2d(input2, filter5, strides=[1, 2, 2, 1], padding='SAME')
vop1 = tf.nn.conv2d(input1, filter1, strides=[1, 2, 2, 1], padding='VALID')

op6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='SAME')
vop6 = tf.nn.conv2d(input3, filter1, strides=[1, 2, 2, 1], padding='VALID')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print('vop1', sess.run([vop6, input3, filter1]))