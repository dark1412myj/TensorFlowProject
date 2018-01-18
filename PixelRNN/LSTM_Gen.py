import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt

HIDE_LAYER = 500
CORLOR_CNT = 167
BITCH_SIZE = 50
STEP_SIZE = 30
IMAGE_HIGH = 20
IMAGE_WIDTH = 20
IMAGE_SIZE = 400

def getcolormap():
    with open('colormap.txt') as file:
        lines = file.readlines()
        color_map={}
        for (i,line) in enumerate(lines):
            ans = re.split('\\D', line.strip())
            ans = [ int(ans[i]) for i in range(3) ]
            color_map[i]=ans
        #print(color_map)
        return color_map

g_color_map = getcolormap()

def show_img(im,colormap):
    #color = getcolormap()
    img = np.array(im)
    img = img.reshape([IMAGE_HIGH, IMAGE_WIDTH])
    out=[]
    for i in range(len(img)):
        out.append([])
        for j in range(len(img[i])):
            out[i].append([])
            out[i][j]=colormap[ img[i][j] ]
    print(np.array(out))
    plt.imshow(np.array(out,dtype=np.ubyte))
    plt.show()

def getbitch(bitch_size,step_size):
    bitch_list=[]
    x = []
    y = []
    with open('pixel_color.txt') as file:
        lines = file.readlines()
        for line in lines:
            ans = re.split('\\D',line.strip())
            ans = [int(ans[i]) for i in range(len(ans))]
            #print(len(ans))
            #show_img( ans ,g_color_map)
            #return
            start = 0
            while start+step_size+1 < len(ans):
                core = ans[start:start+step_size]
                x.append(core)
                exp = ans[start+1:start+step_size+1]
                y.append(exp)
                start+=3
                if len(x)==bitch_size:
                    #print('x',x)
                    #print('y',y)
                    bitch_list.append(x)
                    bitch_list.append(y)
                    x=[]
                    y=[]
    return bitch_list


class PixelRNN(object):
    def add_embedding(self):
        self.embedding = tf.Variable(tf.random_uniform((self.color_size,self.cell_size)), dtype=tf.float32)
        self.embed = tf.nn.embedding_lookup(self.embedding,self.xs)

    def add_input(self):
        self.xs = tf.placeholder(tf.int64, [None, None])
        self.xy = tf.placeholder(tf.int64, [None, None])
        self.global_step=tf.Variable(0,trainable=False)

    def add_cell(self):
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,forget_bias=1.0, state_is_tuple=True)
        self.lstm_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_cell,0.5)
        self.lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.lstm_cell]*2)
        self.init_state = self.lstm_cell.zero_state(self.bitch_size,dtype=tf.float32)
        self.out,self.stat = tf.nn.dynamic_rnn(self.lstm_cell,self.embed,initial_state=self.init_state,time_major=False,dtype=tf.float32)

    def add_output(self):
        #print('out',tf.shape(self.out))
        #den = tf.layers.dense(self.cell_size,self.color_size)
        tmp=[]
        for i in range(STEP_SIZE):
            if i == 0:
                tmp.append( tf.layers.dense(self.out[:,i,:],self.color_size,name='aaa',reuse=False) )
            else:
                tmp.append(tf.layers.dense(self.out[:, i, :], self.color_size,name='aaa',reuse=True))
        self.ans = tf.stack(tmp,1)
        #self.ans = tf.contrib.layers.fully_connected(self.out, self.color_size, activation_fn=None,
        #                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        #                                  biases_initializer=tf.truncated_normal_initializer(stddev=0.1))
        #self.ans = tf.layers.dense(self.out,self.color_size,)
        #print('self.ans',self.ans.shape)
        self.haha = tf.argmax(self.ans,axis=2)
        self.accrupt = tf.reduce_mean( tf.cast(tf.equal(self.haha,self.xy),tf.float32) )
        #print('self.haha',self.haha.shape,'out',self.xy.shape)
        self.shape = tf.shape(self.ans)
        #exit(0)
        self.loss = tf.contrib.seq2seq.sequence_loss(self.ans,self.xy, tf.ones([ self.shape[0] , self.shape[1] ]))
        optimizer = tf.train.AdamOptimizer(0.01)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients,global_step=self.global_step)

    def __init__(self,colorsize,cell_size,bitch_size):
        self.color_size = colorsize
        self.cell_size = cell_size
        self.bitch_size = bitch_size
        #self.input_size = input_size
        #self.output_size = output_size
        #self.step_size = step_size
        with tf.name_scope("input"):
            self.add_input()
        with tf.name_scope("embedding"):
            self.add_embedding()
        with tf.name_scope("lstm"):
            self.add_cell()
        with tf.name_scope("output"):
            self.add_output()


def train():
    rnn = PixelRNN(CORLOR_CNT, HIDE_LAYER,BITCH_SIZE)
    ls = getbitch(BITCH_SIZE, STEP_SIZE)
    print(len(ls))
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # retrain with last

        #ckpt = tf.train.get_checkpoint_state('./rnnmodel/')
        #if ckpt and ckpt.model_checkpoint_path:
         #   print(ckpt.model_checkpoint_path)
         #   saver.restore(sess, ckpt.model_checkpoint_path)

        while True:
            for i in range(len(ls) // 2):
                loss, _,rate = sess.run([rnn.loss, rnn.train_op,rnn.accrupt], feed_dict={rnn.xs: ls[ 2 * i ], rnn.xy: ls[ i*2 + 1]})
                if i % 100 == 0:
                    saver.save(sess, './rnnmodel/data.ckpt', rnn.global_step)
                    print('loss', loss,'acc=',rate)


def test():
    rnn = PixelRNN(CORLOR_CNT, HIDE_LAYER,1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state('./rnnmodel/')
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        gen = [0]
        state = sess.run(rnn.init_state,feed_dict={rnn.xs:[[0]]})
        #print(state)
        #out, state = sess.run([rnn.out, rnn.stat], feed_dict={rnn.xs: [[0]], rnn.init_state: state})
        for i in range(400):
            out,state = sess.run( [rnn.ans,rnn.stat],feed_dict={rnn.xs:[gen[-30:]],rnn.init_state:state})
            #out, state = sess.run([rnn.ans, rnn.stat], feed_dict={rnn.xs: [[0]], rnn.init_state: state})
            print( out[0][-1] )
            pos = np.ndarray.argmax(out[0][-1])
            #print('len = ',len(out[0][0]))
            #print('pos = ',pos)
            gen.append(pos)
            #pos  =  np.ndarray.argmax(out[0][0])
            #print(out[0][0][pos])
        print(gen)
        #out,state = tf.nn.dynamic_rnn(rnn.lstm_cell,rnn.embed, initial_state=state)
        #sess.run([out, state], feed_dict={rnn.xs: [[1]]})
        #for i in range(400):
            #sess.run([rnn.out,rnn.])
            #sess.run([out,state],feed_dict={rnn.xs:[[1]]})


def main(argv=None):
    #getbitch(100,100)
    #colormap = getcolormap()
    #while True:
    train()
    #test()



if __name__ == "__main__":
    tf.app.run()