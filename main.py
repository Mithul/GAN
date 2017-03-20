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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
Z = tf.placeholder(tf.float32, shape=[None, 100])

Wg1 = tf.Variable(xavier_init([100,768]))
bg1=tf.Variable(xavier_init([768]))
Wg2 = tf.Variable(xavier_init([768,3072]))
bg2=tf.Variable(xavier_init([3072]))

theta_G = [Wg1, Wg2, bg1, bg2]
	
W1 = tf.Variable(xavier_init([3072, 768]))
b1=tf.Variable(xavier_init([768]))
W3 = tf.Variable(xavier_init([768,1]))
b3=tf.Variable(xavier_init([1]))
W5 = tf.Variable(xavier_init([192, 64]))
b5=tf.Variable(xavier_init([64]))
W7 = tf.Variable(xavier_init([64, 1]))
b7=tf.Variable(xavier_init([1]))

theta_D = [W1, W3, W5, W7, b1, b3, b5, b7]

def generator(input):
	with tf.variable_scope("generator") as scope:
		layer1 = tf.nn.relu(tf.matmul(input, Wg1) + bg1)
		output = tf.reshape(tf.nn.sigmoid(tf.matmul(layer1, Wg2) + bg2), [-1, 32, 32, 3])
	return output

def discriminator(input):
	with tf.variable_scope("discriminator") as scope:
		input = tf.reshape(input, [-1, 3072])
		layer1 = tf.nn.tanh(tf.matmul(input, W1)+ b1)

		layer3 = tf.matmul(layer1, W3)+ b3

		# layer5 = tf.nn.relu(tf.matmul(layer3, W5)+ b5)

		# layer7 = tf.nn.relu(tf.matmul(layer5, W7)+ b7)

		output = tf.nn.sigmoid(layer3)
		logits = layer3
	return output, logits


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        # print sample.shape
        # plt.imshow(sample.reshape(32, 32, 3), cmap='Greys_r')
        plt.imshow(sample.reshape(32, 32, 3))

    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real+1e-30) + tf.log(1. - D_fake + 1e-30))
# G_loss = -tf.reduce_mean(tf.log(D_fake+1e-30))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 100
Z_dim = 100

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

model_save_dir = "models/"
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(model_save_dir)
if ckpt and ckpt.model_checkpoint_path:
	print("Checkpoint Found ")
	saver.restore(session, ckpt.model_checkpoint_path)

def train():
	i = 0
	for it in range(1000000):
	    if it % 100 == 0:
	        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

	        fig = plot(samples)
	        plt.savefig('out/{}.png'.format(str(i).zfill(6)), bbox_inches='tight')
	        i += 1
	        plt.close(fig)
	    # X_mb, _ = mnist.train.next_batch(mb_size)
	    b = it%(len(train_data)/mb_size)
	    X_mb = train_data[(b)*mb_size:(b+1)*mb_size,:,:,:]
	    if len(X_mb)!=mb_size:
	    	continue
	    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
	    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

	    if it % 1000 == 0:
	        print('Iter: {}'.format(it))
	        print('D loss: {:.4}'. format(D_loss_curr))
	        print('G_loss: {:.4}'.format(G_loss_curr))
	        print()
	        saver.save(sess, model_save_dir + 'model.ckpt',
                           global_step=it+1)

train()