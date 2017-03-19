def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

train_data = unpickle("/home/mithul/Datasets/cifar-100-python/train")

import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


train_data = train_data['data'].reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float")/255


# print train_data[0]
# # cv2.imshow('',train_data[0])
# # cv2.waitKey()
# plt.imshow(train_data[0])
# plt.show()


import tensorflow as tf
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


Wg1 = tf.Variable(xavier_init([100,768]))
bg1=tf.Variable(xavier_init([768]))
Wg2 = tf.Variable(xavier_init([768,3072]))
bg2=tf.Variable(xavier_init([3072]))
theta_D = [Wg1, Wg2, bg1, bg2]
def generator(input):
	with tf.variable_scope("generator") as scope:
		layer1 = tf.nn.relu(tf.matmul(input, Wg1) + bg1)
		output = tf.reshape(tf.nn.sigmoid(tf.matmul(layer1, Wg2) + bg2), [-1, 32, 32, 3])
	return output

W1 = tf.Variable(tf.random_normal([32, 32, 3, 16]))
b1=tf.Variable(tf.random_normal([16]))
W3 = tf.Variable(tf.random_normal([13, 13, 16, 32]))
b3=tf.Variable(tf.random_normal([32]))
W5 = tf.Variable(tf.random_normal([4, 4, 32, 64]))
b5=tf.Variable(tf.random_normal([64]))
W7 = tf.Variable(tf.random_normal([64, 1]))
b7=tf.Variable(tf.random_normal([1]))

theta_G = [W1, W3, W5, W7, b1, b3, b5, b7]
def discriminator(input):
	with tf.variable_scope("discriminator") as scope:
		input = tf.reshape(input, [-1, 32, 32, 3])
		layer1 = tf.nn.conv2d(input, W1, strides=[1,1,1,1], padding='SAME')+ b1
		layer2 = tf.nn.tanh(tf.nn.max_pool(layer1, [1,7,7,1], strides=[1,2,2,1], padding='VALID'))
		
		layer3 = tf.nn.conv2d(layer2, W3, strides=[1,1,1,1], padding='SAME')+ b3
		layer4 = tf.nn.tanh(tf.nn.max_pool(layer3, [1,7,7,1], strides=[1,2,2,1], padding='VALID'))

		layer5 = tf.nn.conv2d(layer4, W5, strides=[1,1,1,1], padding='SAME')+ b5
		layer6 = tf.nn.relu(tf.nn.max_pool(layer5, [1,4,4,1], strides=[1,2,2,1], padding='VALID'))


		# layer5 = tf.nn.relu(tf.matmul(layer3, W5)+ b5)

		layer7 = tf.nn.relu(tf.matmul(tf.reshape(layer6, [-1, 64]), W7)+ b7)

		output = tf.nn.sigmoid(layer7)
		logits = layer7
	return output, logits

input_d = tf.placeholder(tf.float32)
label = tf.placeholder(tf.int8)
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
gen = generator(Z)
d_r, di_r = discriminator(input_d) 
d_f, di_f = discriminator(gen)

g_loss = -tf.reduce_mean(tf.log(d_f))
d_loss = -tf.reduce_mean(tf.log(d_r) + tf.log(1.0-d_f+1e-20))

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=di_r, labels=tf.ones_like(di_r)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=di_f, labels=tf.zeros_like(di_f)))
D_loss = D_loss_real + D_loss_fake
G_loss = -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=di_f, labels=tf.ones_like(di_f)))

d_solver = tf.train.AdamOptimizer(1e-3).minimize(D_loss, var_list=theta_D)
g_solver = tf.train.AdamOptimizer(1e-3).minimize(G_loss, var_list=theta_G)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 10


def train():
	for k in range(1000):
		for i in range(len(train_data)/batch_size):
			sess.run(g_solver, feed_dict={Z: np.random.uniform(-1., 1., size=[batch_size, 100])})
			sess.run(d_solver, feed_dict={Z: np.random.uniform(-1., 1., size=[batch_size, 100]), input_d: train_data[(i)*batch_size:(i+1)*batch_size]})
			if i%1000 == 0:
				show(k*10000+i)
				print sess.run([d_r, d_f, di_r, di_f, G_loss, D_loss], feed_dict={Z: np.random.uniform(-1., 1., size=[batch_size, 100]), input_d: train_data[(i)*batch_size:(i+1)*batch_size]})
				print str(k)+" : "+str(i)+"/"+str(len(train_data)/batch_size)

def show(i):
	plt.imshow(sess.run(gen, feed_dict={Z: np.random.uniform(-1., 1., size=[batch_size, 100])})[6])
	plt.savefig('out/{}.png'.format(str(i).zfill(10)), bbox_inches='tight')
	# plt.show(block=False)
