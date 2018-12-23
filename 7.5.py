# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf


learning_rate = 0.001
traing_epochs = 25
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)
add_global = global_step.assign_add(1)

def multilayer_perception(x, weights, biases, keep_prob):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    keep_layer_1 = tf.nn.dropout(layer_1, keep_prob)

    layer_2 = tf.add(tf.matmul(keep_layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    outlayer = tf.matmul(layer_2, weights['out']) + biases['out']
    return outlayer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

pred = multilayer_perception(x, weights, biases, keep_prob)

reg = 0.01
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) + reg * (tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']))
decay_rate = tf.train.exponential_decay(learning_rate, global_step, 8000, 0.9)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(decay_rate).minimize(cost, global_step=global_step)


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

training_epochs = 400
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
            _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch, keep_prob: 0.7})
            p = sess.run([pred, add_global], feed_dict={x: x_batch, keep_prob: 1.0})
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('Epoch', '%04d' % (epoch+1), 'cost =', avg_cost, 'accu =', accurary.eval({x: x_test, y: y_test, keep_prob: 1.0}))
        saver.save(sess, model_path)
    print('Fnished!')