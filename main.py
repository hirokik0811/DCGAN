'''
Created on Oct 26, 2017

@author: kwibu
'''
import numpy as np
import tensorflow as tf
from dcgan import dcgan
from traindataset import dataset
import cv2
BATCH_SIZE = 128

data = np.load('./Aberdeen_data')
traindata = dataset(data)


DCGAN = dcgan(batch_size=BATCH_SIZE, s_size=4, im_size=64)

input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 64, 64, 3])
losses = DCGAN.loss(traindata=input)
train_op = DCGAN.train(losses)
sample = DCGAN.generate_sample(1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, "./checkpoint/ckpt")
for i in range(10000):
    input_ph = traindata.next_batch(BATCH_SIZE)
    _, g_loss, d_loss = sess.run([train_op, losses["g_loss"], losses["d_loss"]],
                                 feed_dict={input: input_ph})
    if i%100==0:
        print("%d steps --- g_loss: %d, d_loss: %d"%(i, g_loss, d_loss))
        saver.save(sess, "./checkpoint/ckpt")
        images = sess.run([sample])
        for img in images[0]:
            cv2.imshow("", img)
            cv2.waitKey(1000)
            #cv2.imwrite("./poke_result/ckpt%d.jpg"%i, img)
        cv2.destroyAllWindows()