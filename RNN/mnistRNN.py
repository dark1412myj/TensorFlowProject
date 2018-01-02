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


def main(argv=None):
    x = tf.placeholder(tf.float32,[None,28*28],name='x-input')
    y_= tf.placeholder(tf.float32,[None,10],name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    y = inference(x,regularizer)
    global_step = tf.Variable(0,trainable=False)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    loss = cross_entropy #+tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss,global_step=global_step)
    accou = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32 ))
    #tf.keras.layers
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(1000):
            xs,ys = mnist.train.next_batch(BITCH_SIZE)
            #print(xs.shape,'----',ys.shape)
            sess.run(train_step,feed_dict={x:xs,y_:ys})
            if i %100==0:
                print(sess.run(accou,feed_dict={x:xs,y_:ys}))



if __name__ == "__main__":
    x = np.array([[1,2,3],[4,5]])
    y = np.array([[1, 2, 3], [4, 5,6]])
    print(x.shape)
    print(y.shape)
    #tf.app.run()
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
    #print(lstm_cell.state_size)
    #x = 10