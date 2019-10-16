import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data as Cd


max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="cifar-10-batches-bin/"

def variable_with_weight_loss(shape,stddev,wl):
    # 使用 truncated_normal() 函数创建权重参数
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        #这边调用l2损失函数，相当于L2正则化
        weights_loss=tf.multiply(tf.nn.l2_loss(var),wl,name="weight_loss")
        tf.add_to_collection("losses",weights_loss)
    return var
#接下来，生成训练集数据batch和测试集数据batch
#这是用于训练的图片数据，distorted设置为True,表示进行数据增强
images_train,labels_train=Cd.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
#这是用于测试的图片数据
images_test,labels_test=Cd.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)
#下面定义填充张量x,y_
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.int32,[batch_size])

#定义两层卷积层（包括池化）和3层全连接层
#第一层卷积，卷积核：5*5,3个通道，64个卷积核
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev= 5e-2,wl=0.0)

conv1=tf.nn.conv2d(x,kernel1,strides=[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#第二层
kernel2 = variable_with_weight_loss(shape=[5,5,64,64] , stddev=5e-2 , wl=0.0)
conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2 = tf.nn.max_pool(relu2,ksize = [1,3,3,1],strides=[1,2,2,1],padding="SAME")

#接下来是全连接层

#先拉直数据
reshape=tf.reshape(pool2,[batch_size,-1])
dim=reshape.get_shape()[1].value

#第一个全连接层,有384个节点###
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,wl=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,wl=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#第三个全连接层
weight3=variable_with_weight_loss(shape=[192,10],stddev=1/192.0,wl=0.0)
fc_bias3=tf.Variable(tf.constant(0.0,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#整个网络的前向传播完成，为了计算损失值，使用交叉熵+L2正则化
#计算损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))
weight2_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weight2_with_l2_loss
train_op=tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
# 用来计算输出结果的 top k 的准确率, 默认top=1
top_k_op=tf.nn.in_top_k(result,y_,1)

#最后就是创建会话进行训练了


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 开启多线程
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])

        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})

        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            print("step %d ,loss = %.2f (%.1f examples/sec; %.3fsec/batch" % (
            step, loss_value, examples_per_sec, sec_per_batch))

    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size

    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
    true_count += np.sum(predictions)
print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))




