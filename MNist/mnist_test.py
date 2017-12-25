# -*- coding:GBK -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

def Init():
    mnist = input_data.read_data_sets("./data/",one_hot=True)
    return mnist


def get_matrix(shape,regularizer):
    weights = tf.get_variable('weights',shape,initializer=tf.random_normal_initializer(stddev=0.1)) #用initializer的方式不用指定shape
    #weights = tf.Variable(tf.random_normal(shape,stddev=0.1),name='weights')
    if regularizer!= None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights


def inference(input_x,regularizer):
    with tf.variable_scope('layer1'):
        weighte1 = get_matrix([28*28,500],regularizer)
        bais1 = tf.get_variable('bais',[500],initializer=tf.random_normal_initializer(stddev=0.1))
        hide1 = tf.nn.relu( tf.matmul(input_x , weighte1) + bais1 )

    with tf.variable_scope('layer2'):
        weighte2 = get_matrix([500,10],regularizer)
        bais2 = tf.get_variable('bais',[10],initializer=tf.random_normal_initializer(stddev=0.1))
        hide2 = tf.matmul(hide1,weighte2) + bais2        #得到输出层时不需要再做relu

    return hide2


def train(mnist):
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    x = tf.placeholder(tf.float32, [None, 28 * 28], 'input-x')
    y_ = tf.placeholder(tf.float32, [None, 10], 'input-y')
    global_step = tf.Variable(0,trainable=False)
    y = inference(x,regularizer)
    var_averages = tf.train.ExponentialMovingAverage(0.9999,global_step)
    var_averages_op = var_averages.apply(tf.trainable_variables())
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy,global_step=global_step)
    with tf.control_dependencies([train_step,var_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(30000):
            xs,ys = mnist.train.next_batch(100)
            _,loss_value,step = sess.run( [train_op,cross_entropy,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print(loss_value,'----:',step)
                saver.save(sess,os.path.join('./tfmodel/','test.ckpt'),global_step=global_step)


def main(argv=None):
    mnist = Init()
    #train(mnist)
    print(os.path.join('./tfmodel/','test.ckpt'))


if __name__ == "__main__":
    tf.app.run()
