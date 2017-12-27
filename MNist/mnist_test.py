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
    y = inference(x, regularizer)
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
    total_loss = cross_entropy + tf.get_collection('losses')
    train_step = tf.train.AdagradOptimizer(0.1).minimize(total_loss,global_step=global_step)

    correct_mat = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_rate = tf.reduce_mean(tf.cast(correct_mat, tf.float32))

    #with tf.control_dependencies([train_step,var_averages_op]):
    #    train_op = tf.no_op(name='train')
    with tf.control_dependencies([train_step]):#定义一些列的动作，相当于绑定一些操作在一起
        train_op = tf.no_op(name='train')      #定义空操作，因为该操作以来前面的，所以相当于前几个动作的连续执行

    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(30000):
            xs,ys = mnist.train.next_batch(100)
            _,loss_value,step = sess.run( [train_op,cross_entropy,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print(loss_value,'----:',step,sess.run(correct_rate,feed_dict={x:xs,y_:ys}))
                saver.save(sess,os.path.join('./tfmodel/','test.ckpt'),global_step=global_step)


def evaluate(mnist):
    x = tf.placeholder(tf.float32, [None, 28 * 28], 'input-x')
    y_ = tf.placeholder(tf.float32, [None, 10], 'input-y')
    global_step = tf.Variable(0, trainable=False)
    y = inference(x, None)
    saver = tf.train.Saver()
    correct_mat = tf.equal( tf.argmax(y,1),tf.argmax(y_,1) )
    correct_rate = tf.reduce_mean( tf.cast(correct_mat,tf.float32) )
    with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state('./tfmodel/')
        if ckpt and ckpt.model_checkpoint_path:
            for i_path in ckpt.all_model_checkpoint_paths:
                saver.restore(sess,i_path)
                global_step2 = i_path.split('/')[-1].split('-')[-1]
                print('global_step=',global_step2,':',sess.run(correct_rate,feed_dict={x:mnist.test.images,
                                                     y_:mnist.test.labels})*100,'%')


def main(argv=None):
    mnist = Init()
    #train(mnist)
    evaluate(mnist)
    #print(os.path.join('./tfmodel/','test.ckpt'))


if __name__ == "__main__":
    tf.app.run()
