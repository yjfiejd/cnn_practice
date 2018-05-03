from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#【1】定义计算准确度函数
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

#【2】定义weight_variable, bias_variable. conv2d, max_poll_2x2变量，并初始化
def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    #stride[1,x_movement, y_movement,1]
    return  tf.nn.conv2d(x, W, strides=[1,1,1,1], padding= 'SAME') #左右跨1步

def max_poll_2x2(x):
    # stride[1,x_movement, y_movement,1] 第一位和第四位一致
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #左右跨2步

#【3】类似于定义形参 xs, ys
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1]) #这里图片因为是黑白的，高度只有1

#【4】定义
##conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32]) #patch 5*5, insize 1, outsize 32, 用5*5的小方块扫描，传入的高度为1，输出的高度为32
b_conv1 = bias_variable(([32]))
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #ouputsize: 28*28*32, 这里图片长宽不变，因为用的是same padding
h_pool1 = max_poll_2x2(h_conv1)                          #outsize: 14*14*32 ，这里pooling 后长宽变成了14*14
#第一层后，长宽变为了14*14，高度为32

#conv2 layer,
W_conv2 = weight_variable([5, 5, 32, 64]) #patch 5*5, insize 32, outsize 64 , 用5*5的小方块扫描，传入的为32，让它输出为64的高度
b_conv2 = bias_variable(([64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #ouputsize: 14*14*64, 这里图片长宽不变，因为用的是same padding
h_pool2 = max_poll_2x2(h_conv2)                          #outsize: 7*7*64
#第二层后，长宽变为了7*7，高度为64

#func1 layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
#[n_sample, 7,7,64] --> 变成1个维度的[n_sample, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#func2 layer
W_fc2 = weight_variable([1024, 10]) #输入1024高度，最后需要输出的10的高度，用来分类
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


#error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
#初始化
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

#输出
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))



#最后的输出结果：准确率
# 0.104
# 0.763
# 0.853
# 0.891
# 0.903
# 0.908
# 0.913
# 0.931
# 0.927
# 0.932
# 0.929
# 0.934
# 0.944
# 0.946
# 0.945
# 0.954
# 0.953
# 0.954
# 0.958
# 0.952