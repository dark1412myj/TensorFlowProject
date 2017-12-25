import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random


def int2vector(i):
    ele=[]
    dim = 10
    while (dim != 0):
        dim = dim - 1
        ele.append(i % 2)
        i = i // 2
    return ele


def getLabel(x):
    if x % 15 == 0:
        return [0,0,0,1]
    if x % 5 == 0:
        return [0,0,1,0]
    if x % 3 == 0:
        return [0,1,0,0]
    return [1,0,0,0]


def getTranData(x,y,start=10,end=900):
    for i in range(start,end):

        y.append(getLabel(i))

        x.append(int2vector(i))


def train():
    data=[]
    label=[]
    getTranData(data,label)
    x = tf.placeholder(tf.float32,[None,10],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,4],name='y-output')
    weight1 = tf.Variable(tf.random_normal([10,1000],stddev=0.1) )
    bias1 = tf.Variable(tf.zeros([1000]))

    weight2 = tf.Variable(tf.random_normal([1000, 4], stddev=0.1))
    bias2 = tf.Variable(tf.zeros([4]))

    hide = tf.nn.relu( tf.matmul( x,weight1 ) + bias1 )
    y = tf.matmul(hide,weight2)+bias2

    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(0.1,global_step,2,0.999999,staircase=False)
    loss = tf.reduce_sum( tf.square(y-y_) )
    loss2 =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
    #cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2,global_step=global_step)
    #train_step = tf.train.AdagradOptimizer( 0.1 ).minimize(loss2, global_step=global_step)
    correct_prediction = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(50000):
            start = random.randint(0,len(label)-1 )
            sess.run(train_step,feed_dict={x:data[start:],y_:label[start:]})
            if i % 1000 == 0:
                print( sess.run( loss2,feed_dict={x:data,y_:label} ))
                print(sess.run(accuracy, feed_dict={x: data, y_: label})*100,'%')
        xx =[]
        xx.append(int2vector(1000))
        print( sess.run(y,feed_dict={x:xx}) )
        saver = tf.train.Saver()
        saver.save(sess,"./data.ckpt")


def test():

    x = tf.placeholder(tf.float32, [None, 10], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 4], name='y-output')
    weight1 = tf.Variable(tf.random_normal([10, 1000], stddev=0.1))
    bias1 = tf.Variable(tf.zeros([1000]))

    weight2 = tf.Variable(tf.random_normal([1000, 4], stddev=0.1))
    bias2 = tf.Variable(tf.zeros([4]))

    hide = tf.nn.relu(tf.matmul(x, weight1) + bias1)
    y = tf.matmul(hide, weight2) + bias2

    global_step = tf.Variable(0, trainable=False)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,"./data.ckpt")
        test_data = []
        test_label = []
        getTranData(test_data,test_label,1000,1024)
        print(sess.run(accuracy, feed_dict={x: test_data,y_:test_label}))
        print(global_step.name,global_step.eval())

if __name__ == "__main__":
    #train()
    test()

