# -*- coding:GBK -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./data/",one_hot=True)

BITCH_SIZE = 100
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1

def inferenct(input_x,regularizer):
    with tf.variable_scope('conv1'):#第一层为卷基层
        weight = tf.get_variable('weight',[5,5,1,10],initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[10],initializer=tf.random_normal_initializer(stddev=0.1))
        hide1 = tf.nn.relu(tf.nn.bias_add( tf.nn.conv2d(input_x,weight,strides=[1,1,1,1]) , bias) )
    with tf.variable_scope('pool1'):#第二层为池化层
        hide2 = tf.nn.max_pool(hide1,ksize=[1,2,2,1],strides=[1,2,2,1])
    with tf.variable_scope('conv2'):
        weight = tf.get_variable('weight',[5,5,10,64],initializer=tf.random_normal_initializer(stddev=0.1))
        bias = tf.get_variable('bias',[64],initializer=tf.random_normal_initializer(stddev=0.1))
        hide3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(hide2,weight,strides=[1,1,1,1]),bias))
    with tf.variable_scope('pool2'):
        hide4 = tf.nn.max_pool(hide3,ksize=[1,2,2,1],strides=[1,2,2,1])

def train():
    x = tf.placeholder(tf.float32,[BITCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL],name='input-x')
    y_ = tf.placeholder(tf.float32,[BITCH_SIZE,10],name='output-y_')
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y = inferenct(x,regularizer)

def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()