'''
Created on Oct 20, 2017

@author: kwibu
'''
import tensorflow as tf
import numpy as np

class generator:
    def __init__(self, n_filters = [1024, 512, 256, 128], first_fsc_size=4, z_dim=100):
        self.first_fsc_size = first_fsc_size
        self.n_filters = n_filters+[3]
        self.z_dim = z_dim
        self.reuse = False
    
    def __call__(self, input):
        """
        return generated images from randomized distributions.
        input: batch_size*INPUT_Z
        output: batch_size*im_size*im_size*3(RGB)
        """
        out = tf.convert_to_tensor(input)
          
        with tf.variable_scope("gen", reuse = self.reuse):
            def weight_variable(shape, name):
                return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            def bias_variable(shape, name):
                return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer(tf.float32))
            with tf.variable_scope("fully"):
                w_fc = weight_variable([self.z_dim, self.first_fsc_size**2*self.n_filters[0]], name="weight_fc")
                b_fc = bias_variable([self.n_filters[0]], name="bias_fc")
                fc1 = tf.reshape(tf.matmul(out, w_fc), [-1, self.first_fsc_size, self.first_fsc_size, self.n_filters[0]])+b_fc
                mean0, variance0 = tf.nn.moments(fc1, [0, 1, 2])
                out = tf.nn.relu(tf.nn.batch_normalization(fc1, mean0, variance0, None, None, 1e-5))
                
            for i in range(len(self.n_filters)-1):
                with tf.variable_scope("conv%d" % i):
                    w_fsc = weight_variable([5, 5, self.n_filters[i+1], self.n_filters[i]], name="weight_conv%d"%i)
                    b_fsc = bias_variable([self.n_filters[i+1]], name="bias_conv%d"%i)
                    fsc = tf.nn.conv2d_transpose(out, w_fsc,
                        [tf.shape(out)[0], self.first_fsc_size*2**(i+1), self.first_fsc_size*2**(i+1), self.n_filters[i+1]],
                        strides=[1, 2, 2, 1])
                    out = fsc + b_fsc
                    if (i == len(self.n_filters)-2):
                        out = tf.nn.tanh(out)
                    else:
                        mean, variance = tf.nn.moments(fsc, [0, 1, 2])
                        out = tf.nn.relu(tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5))
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="gen")
        return out

class discriminator:
    def __init__(self, n_filters = [128, 256, 512, 1024], last_sc_size = 4, im_size=64):
        self.n_filters = [3] + n_filters
        self.last_sc_size = last_sc_size
        self.im_size = im_size
        self.reuse = False
        
    def __call__(self, input):
        """
        return the result of discrimination to given inputs
        (if the given input is generated or not)
        input: batch_size*im_size*im_size*3(RGB)
        output: onehot(2)
        """
        out = tf.convert_to_tensor(input)
        with tf.variable_scope("disc", reuse=self.reuse):
            def weight_variable(shape, name):
                return tf.get_variable(name, shape, tf.float32, tf.truncated_normal_initializer(stddev=0.02))
            def bias_variable(shape, name):
                return tf.get_variable(name, shape, tf.float32, tf.zeros_initializer(tf.float32))
            for i in range(len(self.n_filters)-1):
                with tf.variable_scope("conv%d" % i):
                    w_fsc = weight_variable([5, 5, self.n_filters[i], self.n_filters[i+1]], name="weight_conv%d"%i)
                    b_fsc = bias_variable([self.n_filters[i+1]], name="bias_conv%d"%i)
                    fsc = tf.nn.conv2d(out, w_fsc, padding='SAME', strides = [1, 2, 2, 1])
                    out = fsc + b_fsc
                    mean, variance = tf.nn.moments(fsc, [0, 1, 2])
                    out = tf.nn.batch_normalization(out, mean, variance, None, None, 1e-5)
                    out = tf.maximum(0.2*out, out) #leakyRelu
            with tf.variable_scope("fully"):
                last_sc_size = int(self.im_size*(1/2**(len(self.n_filters)-1)))
                out = tf.reshape(out, [-1, last_sc_size*last_sc_size*self.n_filters[-1]])
                w_fc = weight_variable([last_sc_size*last_sc_size*self.n_filters[-1], 2], name="weight_fc")
                b_fc = bias_variable([2], name="bias_fc")
                out = tf.matmul(out, w_fc)+b_fc
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="disc")
        return out
    
class dcgan:
    def __init__(self, batch_size=128, s_size=4, z_dim=100, im_size=64,
                 g_n_filters = [512, 256, 128, 64],
                 d_n_filters = [64, 128, 256, 512]):
        self.batch_size = batch_size
        self.s_size = s_size
        self.gen = generator(g_n_filters, s_size, z_dim)
        self.disc = discriminator(d_n_filters, s_size, im_size)
        self.z_dim = z_dim
        self.z = tf.random_normal([batch_size, z_dim], mean=0, stddev=0.02, name="z")
        
    def loss(self, traindata):
        """
        build models and calculate losses.
        traindata: batch_size*im_size*im_size
        """
        generated = self.gen(self.z)
        gen_out = self.disc(generated)
        train_out = self.disc(traindata)
        tf.add_to_collection(
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=gen_out)))

        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=train_out)))
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=gen_out)))
        return {
            "g_loss": tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            "d_loss": tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }
    def train(self, losses, learning_rate=0.0001, beta1=0.5):
        """
        train models
        """
        g_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_optimizer.minimize(losses["g_loss"],
                                        var_list=self.gen.variables)
        d_opt_op = d_optimizer.minimize(losses["d_loss"],
                                        var_list=self.disc.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name="train")
    
    def generate_sample(self, n_samples=1):
        """
        output generated samples
        """
        z = tf.random_normal(shape=[n_samples, self.z_dim], mean=0, stddev=0.02, name="z")
        generated = self.gen(z)
        converted = tf.image.convert_image_dtype(
            tf.divide(tf.add(generated, 1.0), 2.0), dtype=tf.uint8)
        #images = [tf.image.encode_png(tf.squeeze(image, [0])) for image in tf.split(converted, n_samples, axis=1)]
        images = [tf.squeeze(image, [0]) for image in tf.split(converted, num_or_size_splits=n_samples, axis=0)]
        return images