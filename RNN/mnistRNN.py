import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('./data/',one_hot=True)
BITCH_SIZE = 128
HIDE_LAYER = 100


def inference(x,regularizer):

    x = tf.reshape(x,[-1,28,28])

    lstm_cell = tf.nn.rnn_cell.LSTMCell(HIDE_LAYER)

    out,state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32,time_major=False)

    result = tf.layers.dense(out[:,-1,:],10)
    #print(out.shape,'---',x.shape)
    return result

def train(x,y_):
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y = inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = cross_entropy  # +tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss, global_step=global_step)
    accou = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(1000):
            xs, ys = mnist.train.next_batch(BITCH_SIZE)
            # print(xs.shape,'----',ys.shape)
            sess.run(train_step, feed_dict={x: xs, y_: ys})
            if i % 100 == 0:
                print(sess.run(accou, feed_dict={x: xs, y_: ys}))
                saver.save(sess,'./model/',global_step)

def test(x,y_):
    y = inference(x,None)
    global_step = tf.Variable(0, trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = cross_entropy  # +tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss, global_step=global_step)
    correct_rate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    saver = tf.train.Saver()
    xs, ys = mnist.test.next_batch(BITCH_SIZE)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model/')
        if ckpt and ckpt.model_checkpoint_path:
            for i_path in ckpt.all_model_checkpoint_paths:
                saver.restore(sess, i_path)
                global_step2 = i_path.split('/')[-1].split('-')[-1]
                print('global_step-', global_step2, '-correct rate:',
                      sess.run(correct_rate, feed_dict={x: xs, y_: ys}) * 100, '%')


def main(argv=None):
    x = tf.placeholder(tf.float32,[None,28*28],name='x-input')
    y_= tf.placeholder(tf.float32,[None,10],name='y-input')
    #train(x,y_)
    test(x,y_)


if __name__ == "__main__":
    tf.app.run()
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
    #print(lstm_cell.state_size)
    #x = 10