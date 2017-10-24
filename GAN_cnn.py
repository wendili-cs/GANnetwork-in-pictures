#coding:utf-8
"""
python 3.6
tensorflow 1.3
By LiWenDi
"""
#生成器为全连接层，辨识器使用CNN网络
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path
import cv2
import random

toTrain = False #训练模式/输出模式
toContinueTrain = True #训练模式下继续上次的训练
toShuffle = True #是否打乱顺序
toShow = False #训练途中是否展示当前成果
howManyShow = 5 #每迭代多少次展示一次

new_train = True #是否允许重复训练直至loss达到要求
generator_first = True #优先满足生成器/辨别器的要求
generator_loss_demand = 1.0 #generator的loss要求
discriminator_loss_demand = 1.5 #discriminator的loss要求
max_re = 5 #最大允许重复训练次数

#生成图片
do_times = 10
generate_num = 10

#展示图片
show_batch_num = 4

output_path = "trained_model_others3/model.ckpt"
total_epoch = 1000
batch_size = 4 #决定生成品的特异性，越小越具有特异性
learning_rate_generator = 0.000002
learning_rate_discriminator = 0.000002
n_hidden = 256 #决定样本特征的因子，越大能描述的特征越多
n_width = 64
n_height = 64
n_input = n_width*n_height*3
n_noise = 64 #决定生成品共性，越大越离散

#WGAN参数
w_clip = 0.005

#--------------------读取数据-----------------------
input_data = "sh_p"
image_dirs = os.listdir(input_data)
image_data = []
np.random.shuffle(image_dirs)
#print(image_dirs)
for image_dir in image_dirs:
    image_dir = os.path.join('%s/%s' % (input_data, image_dir))
    #print(image_dir)
    temp_pic = cv2.imread(image_dir,cv2.IMREAD_COLOR)
    b,g,r = cv2.split(temp_pic)  
    img = cv2.merge([r,g,b])
    image_data.append(cv2.resize(img, (n_height, n_width)))
    #image_data.append(cv2.imread(image_dir,cv2.IMREAD_GRAYSCALE))
dataset_num = len(image_data)
image_data = np.array(image_data, float)
image_data = image_data / 255.0

image_data = np.reshape(image_data, [len(image_data), n_height*n_width*3])
#print(image_data.shape)
#----------------------------------------------------

if not toTrain:
    toContinueTrain = False


#-------------------------建立GAN网络----------------------------
#Descriminator网络输入图片形状
x = tf.placeholder(tf.float32,[None,n_input])
x_image = tf.reshape(x,  [-1, n_height, n_width, 3])
#Generator网络输入的是噪声
z = tf.placeholder(tf.float32,[None,n_noise])

#Generator网络的权重和偏置
Generator_param={
    'gw_1':tf.Variable(tf.random_normal([n_noise,n_hidden],stddev=0.1)),
    'gb_1':tf.Variable(tf.zeros([n_hidden])),
    'gw_2':tf.Variable(tf.random_normal([n_hidden,n_input],stddev=0.1)),
    'gb_2':tf.Variable(tf.zeros([n_input]))
}

#-------------------------卷积结构-----------------------------------
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
#-------------------------卷积结构-----------------------------------

#Discriminator网络权重和偏置
n_input_toConv = n_width*n_height * 96 // 4
Discriminator_param={
    'dconv_w1':tf.Variable(tf.truncated_normal([3, 3, 3, 48], stddev = 0.1)),
    'dconv_b1':tf.Variable(tf.zeros(shape = [48])),
    'dconv_w2':tf.Variable(tf.truncated_normal([3, 3, 48, 96], stddev = 0.1)),
    'dconv_b2':tf.Variable(tf.zeros(shape = [96])),
    'dw_1':tf.Variable(tf.random_normal([n_input_toConv,n_hidden],stddev=0.1)),
    'db_1':tf.Variable(tf.zeros([n_hidden])),
    'dw_2':tf.Variable(tf.random_normal([n_hidden,1],stddev=0.1)),
    'db_2':tf.Variable(tf.zeros([1]))
}

#构建Generator网络
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z,Generator_param['gw_1'])+Generator_param['gb_1'])
    output = tf.nn.sigmoid(tf.matmul(hidden,Generator_param['gw_2'])+Generator_param['gb_2'])
    return output

#构建Discriminator网络
def discriminator(inputs):
    h_conv1 = tf.nn.relu(conv2d(tf.reshape(inputs, [-1, n_height, n_width, 3]), Discriminator_param['dconv_w1']) + Discriminator_param['dconv_b1'])
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, Discriminator_param['dconv_w2']) + Discriminator_param['dconv_b2'])
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, n_input_toConv])
    hidden = tf.nn.relu(tf.matmul(h_pool2_flat,Discriminator_param['dw_1'])+Discriminator_param['db_1'])
    output = tf.nn.sigmoid(tf.matmul(hidden,Discriminator_param['dw_2'])+Discriminator_param['db_2'])
    return output
#----------------------------------------------------------------

#生成网络根据噪声生成一张图片
generator_output = generator(z)
#判别网络根据生成网络生成的图片片别其真假概率
discriminator_pred = discriminator(generator_output)
#判别网络根据真实图片判别其真假概率
discriminator_real = discriminator(x)

#生成网络loss
#generator_loss = tf.reduce_mean(tf.log(discriminator_pred))
#判别网络loss
#discriminator_loss = tf.reduce_mean(tf.log(discriminator_real)+tf.log(1 - discriminator_pred))

#生成网络loss
generator_loss = tf.reduce_mean(tf.log(tf.clip_by_value(discriminator_pred, 1e-6, 0.9999)))
#判别网络loss
discriminator_loss = tf.reduce_mean(tf.log(tf.clip_by_value(discriminator_real, 1e-6, 0.9999))+tf.log(1 - tf.clip_by_value(discriminator_pred, 1e-6, 0.9999)))

generator_param_list=[Generator_param['gw_1'],Generator_param['gb_1'],Generator_param['gw_2'],Generator_param['gb_2']]
discriminator_param_list=[Discriminator_param['dconv_w1'],Discriminator_param['dconv_b1'],Discriminator_param['dconv_w2'],Discriminator_param['dconv_b2'],Discriminator_param['dw_1'],Discriminator_param['db_1'],Discriminator_param['dw_2'],Discriminator_param['db_2']]

generator_train = tf.train.RMSPropOptimizer(learning_rate_generator).minimize(-generator_loss,var_list=generator_param_list)
discriminator_train = tf.train.RMSPropOptimizer(learning_rate_discriminator).minimize(-discriminator_loss,var_list=discriminator_param_list)
#截断clip
clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -w_clip, w_clip)) for var in discriminator_param_list]

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    if toContinueTrain:
        saver.restore(sess, output_path)
    else:
        sess.run(init)
    total_batch = int(dataset_num/batch_size)
    generator_c,discriminator_c = 0,0
    #开始交互模式
    #plt.ion()
    if toTrain:
        for epoch in range(total_epoch):
            if toShuffle:
                np.random.shuffle(image_data)
                #print(image_data[0])
            for i in range(total_batch):
                #batch_x,batch_y = mnist.train.next_batch(batch_size)
                batch_x = image_data[i * batch_size: (i + 1) * batch_size]
                #print(i)
                noise = np.random.normal(size=(batch_size,n_noise))
                _,generator_c = sess.run([generator_train,generator_loss],feed_dict={z:noise})
                _,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise})
                sess.run(clip_discriminator_var_op)
                if new_train:
                    if generator_first:
                        now_re = 0
                        while(-discriminator_c > discriminator_loss_demand and now_re <= max_re):
                            _,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise})
                            sess.run(clip_discriminator_var_op)
                            now_re = now_re + 1
                            #print("识别器进行了一次训练")
                        now_re = 0
                        while(-generator_c > generator_loss_demand and now_re <= max_re):
                            _,generator_c = sess.run([generator_train,generator_loss],feed_dict={z:noise})
                            now_re = now_re + 1
                            #print("生成器进行了一次训练")
                    else:
                        now_re = 0
                        while(-generator_c > generator_loss_demand and now_re <= max_re):
                            _,generator_c = sess.run([generator_train,generator_loss],feed_dict={z:noise})
                            now_re = now_re + 1
                        now_re = 0
                        while(-discriminator_c > discriminator_loss_demand and now_re <= max_re):
                            _,discriminator_c = sess.run([discriminator_train,discriminator_loss],feed_dict={x:batch_x,z:noise})
                            sess.run(clip_discriminator_var_op)
                            now_re = now_re + 1
                #print(sess.run(Generator_param['gw_1']))
                #print(sess.run(Discriminator_param['dw_1']))
                #print(error吧)
            if epoch % 10 ==0:
                print('迭代次数: ',int(epoch/10 + 1),'--生成器_loss: %.4f' %-generator_c,'--辨别器_loss: %.4f' %-discriminator_c)
                print("--------------------------------------------------------")
                
                '''
                print("generator_output是：")
                print(sess.run(generator_output,feed_dict={z:noise}))
                print("discriminator_pred是：")
                print(sess.run(discriminator_pred,feed_dict={z:noise}))
                print("discriminator_real是：")
                print(sess.run(discriminator_real,feed_dict={x:batch_x}))
                '''
                saver.save(sess, output_path)

            #图片显示
            if epoch % (10*howManyShow) == 0 and toShow:
                new_batch = show_batch_num
                noise = np.random.normal(size=(new_batch,n_noise))
                #生成图像
                samples = sess.run(generator_output,feed_dict={z:noise})
                fig,a = plt.subplots(1,new_batch,figsize=(new_batch*2,2))
                for i in range(new_batch):
                    a[i].clear()
                    a[i].set_axis_off()
                    a[i].imshow(np.reshape(samples[i],(n_height,n_width,3)))
                plt.draw()
                plt.show()
    else:
        for t in range(do_times):
            saver.restore(sess, output_path)
            new_batch = generate_num
            noise = np.random.normal(size=(new_batch,n_noise))
            #生成图像
            samples = sess.run(generator_output,feed_dict={z:noise})
            fig,a = plt.subplots(1,new_batch,figsize=(new_batch,1))
            for i in range(new_batch):
                a[i].set_axis_off()
                a[i].imshow(np.reshape(samples[i],(n_height,n_width,3)))
            plt.savefig('samples_others3/output_{0}.png'.format(t))
            plt.close(fig)
            print("生成了第{0}幅图片".format(t+1))