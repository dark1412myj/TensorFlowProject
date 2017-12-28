# -*- coding:GBK -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./data/",one_hot=True)

BITCH_SIZE = 100
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1


def inferenct(input_x,regularizer,train):
    with tf.variable_scope('conv1'):#第一层为卷基层
        weight = tf.get_variable('weight',[5,5,1,10],initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[10],initializer=tf.random_normal_initializer(stddev=0.1))
        hide1 = tf.nn.relu(tf.nn.bias_add( tf.nn.conv2d(input_x,weight,strides=[1,1,1,1],padding='VALID') , bias) )
    with tf.variable_scope('pool1'):#第二层为池化层
        hide2 = tf.nn.max_pool(hide1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    with tf.variable_scope('conv2'):
        weight = tf.get_variable('weight',[5,5,10,64],initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[64],initializer=tf.random_normal_initializer(stddev=0.1))
        hide3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(hide2,weight,strides=[1,1,1,1],padding='VALID'),bias))
    with tf.variable_scope('pool2'):
        hide4 = tf.nn.max_pool(hide3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    pool_shape = hide4.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped = tf.reshape(hide4,[pool_shape[0],nodes])
    with tf.variable_scope('conn1'):
        weight = tf.get_variable('weight',[nodes,512],initializer=tf.random_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weight))
        bias = tf.get_variable('bias',[512],initializer=tf.random_normal_initializer(stddev=0.1))
        hide5 = tf.nn.relu( tf.matmul(reshaped,weight) + bias )
        if train:
            hide5_dropout = tf.nn.dropout(hide5,0.5)
        else:
            hide5_dropout = hide5
    with tf.variable_scope('out'):
        weight = tf.get_variable('weight',[512,10],initializer=tf.random_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(weight))
        bias = tf.get_variable('bias',[10],initializer=tf.random_normal_initializer(stddev=0.1))
        hide6 = tf.matmul(hide5_dropout,weight)+bias
    return hide6


def train():
    x = tf.placeholder(tf.float32,[BITCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL],name='input-x')
    y_ = tf.placeholder(tf.float32,[BITCH_SIZE,10],name='output-y_')
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y = inferenct(x, regularizer, True)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_)
    loss = tf.reduce_mean(cross_entropy)+tf.add_n(tf.get_collection('losses'))
    global_step = tf.Variable(0,trainable=False)
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss,global_step=global_step)
    correct_mat = tf.equal( tf.argmax(y,1),tf.argmax(y_,1) )
    correct_rate = tf.reduce_mean(tf.cast(correct_mat,tf.float32))
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(100000):
            xs,ys = mnist.train.next_batch(BITCH_SIZE)
            rex = xs.reshape([BITCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL])
            _,ls = sess.run([train_step,loss],feed_dict={x:rex,y_:ys})
            if(i%1000==0):
                print(ls,sess.run(correct_rate,feed_dict={x:rex,y_:ys}))
                saver.save(sess,'./model/data.ckpt',global_step)


def evaluate():
    x = tf.placeholder(tf.float32, [mnist.test.num_examples, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name='input-x')
    y_ = tf.placeholder(tf.float32, [mnist.test.num_examples, 10], name='output-y_')
    y = inferenct(x, None, False)
    correct_mat = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_rate = tf.reduce_mean(tf.cast(correct_mat, tf.float32))
    saver = tf.train.Saver()
    rex = mnist.test.images.reshape([mnist.test.num_examples,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL])
    ys = mnist.test.labels
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and ckpt.model_checkpoint_path:
            for i_path in ckpt.all_model_checkpoint_paths:
                saver.restore(sess, i_path)
                global_step2 = i_path.split('/')[-1].split('-')[-1]
                print( 'global_step-',global_step2,'-correct rate:',sess.run(correct_rate,feed_dict={x:rex,y_:ys})*100,'%' )


def main(argv=None):
    #train()
    evaluate()


if __name__ == "__main__":
    tf.app.run()