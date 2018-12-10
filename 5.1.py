# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def oneHot(label, c):
    n = label.shape[0]
    b = np.zeros((n, c), dtype=np.int)
    for i in range(n):
        b[i, label[i]] = 1
    return b

mnist = np.load('data/mnist/mnist.npz')
x_train = mnist['x_train'] / 255
y_train = oneHot(mnist['y_train'],10)
x_test = mnist['x_test'] / 255
y_test = oneHot(mnist['y_test'], 10)
x_test = np.reshape(x_test, (x_test.shape[0], -1))

print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)
print('x_test.shape', x_test.shape)
print('y_test.shape', y_test.shape)

tf.reset_default_graph()

inputDict = {
    'x': tf.placeholder(tf.float32, [None, 784]),
    'y': tf.placeholder(tf.float32, [None, 10])
}

paramDict = {
    'w': tf.Variable(tf.random_normal([784, 10])),
    'b': tf.Variable(tf.zeros([10]))
}

pred = tf.nn.softmax(tf.matmul(inputDict['x'], paramDict['w']) + paramDict['b'])
cost = tf.reduce_mean(-tf.reduce_sum(inputDict['y'] * tf.log(pred), reduction_indices=1))

learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


training_epochs = 100
batch_size = 100
display_step = 1
saver = tf.train.Saver()
model_path = 'checkpoint/final.ckpt'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(x_train.shape[0] / batch_size)
        for i in range(total_batch):
            x_batch = np.reshape(x_train[i * batch_size : (i + 1) * batch_size, :], (batch_size, -1))
            y_batch = y_train[i * batch_size : (i + 1) * batch_size]
            _, c = sess.run([optimizer, cost], feed_dict={inputDict['x']: x_batch, inputDict['y']: y_batch})
            p = sess.run(pred, feed_dict={inputDict['x']: x_batch})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(inputDict['y'], 1))
            accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Epoch', '%04d' % (epoch+1), 'cost =', avg_cost, 'accu =', accurary.eval({inputDict['x']: x_test, inputDict['y']: y_test}))
        saver.save(sess, model_path)
    print('Fnished!')