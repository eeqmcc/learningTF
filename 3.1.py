# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w) : idx]) / w for idx, val in enumerate(a)]

train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3

plt.figure(1)
plt.subplot(211)
plt.plot(train_X, train_Y, 'ro', label='Original data')

inputdict = {
    'x': tf.placeholder('float'),
    'y': tf.placeholder('float')
}

paradict = {
    'w': tf.Variable(tf.random_normal([1]), name='weight'),
    'b': tf.Variable(tf.zeros([1]), name='bias')
}

z = tf.multiply(inputdict['x'], paradict['w']) + paradict['b']

cost = tf.reduce_mean(tf.square(inputdict['y'] - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

training_epochs = 20
display_step = 2

with tf.Session() as sess:
    sess.run(init)
    plotdata = {'batchsize':[], 'loss':[], 'avgloss':[]}
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={inputdict['x']:x, inputdict['y']:y})

        if epoch % display_step == 0:
            # loss = sess.run(cost, feed_dict={inputdict['x']:train_X, inputdict['y']:train_Y})
            loss = cost.eval({inputdict['x']:train_X, inputdict['y']:train_Y})
            print('epoch', epoch+1, 'cost =', loss, 'W =', sess.run(paradict['w']), 'b =', sess.run(paradict['b']))
            if not (loss == 'NA'):
                plotdata['batchsize'].append(epoch)
                plotdata['loss'].append(loss)

    print('Finished!')
    print('cost =', loss, 'w =', sess.run(paradict['w']), 'b =', sess.run(paradict['b']))
    plt.plot(train_X, sess.run(paradict['w']) * train_X + sess.run(paradict['b']), 'b+', label='Fitted_data')
    plt.legend()

    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.subplot(212)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('loss')

    print('x = 0.2, z =', sess.run(z, feed_dict={inputdict['x']:0.2}))

plt.show()