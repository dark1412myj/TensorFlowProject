# -*- coding:GBK -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("./data/",one_hot=True)

BITCH_SIZE = 100
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1

def train():
    x = tf.placeholder(tf.float32,[BITCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL],name='input-x')
    y_ = tf.placeholder(tf.float32,[BITCH_SIZE,10],name='output-y_')
def main(argv=None):
    train()

if __name__ == "__main__":
    tf.app.run()