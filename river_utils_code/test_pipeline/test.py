
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
#import gc
import cv2
#import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from os import listdir
from os.path import isfile, join
from datetime import datetime
import glob
import os

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")    #for tensorboard
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
root_logdir_m = "tf_models"
logdir_m = "{}/run-{}/".format(root_logdir_m, now)


# In[2]:


def _parse_function(example_proto):
    
        
        
    features = {
                "image_y": tf.FixedLenFeature((), tf.string ),
                "image_m": tf.FixedLenFeature((), tf.string )
                #"image_x": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)
    image_m = tf.decode_raw(parsed_features["image_m"],  tf.float64)
    
    image_y = tf.reshape(image_y, [256,256,1])
    image_m = tf.reshape(image_m, [256,256,1])
    #image_x = tf.decode_raw(parsed_features["image_x"],  tf.float64)
    #tf.summary.image("64_Y",image_y,3)
    #tf.summary.image("64_M",image_m,3)
    
    #denoise = cv2.fastNlMeansDenoising(image_y, h=24, templateWindowSize=7, searchWindowSize=21)

    image_y = tf.cast(image_y,dtype=tf.float32)
    image_m = tf.cast(image_m,dtype=tf.float32)
    #image_x = tf.cast(image_x,dtype=tf.float32)
    #tf.summary.image("32_Y",image_y,3)
    #tf.summary.image("32_M",image_m,3)
    #image_y = tf.image.total_variation(image_y)

    #image_y = tf.reshape(image_y, [512,512,1])
    #image_m = tf.reshape(image_m, [512,512,1])
    #image_x = tf.reshape(image_x, [512,512,1])

    return image_y,image_m
    #return denoise,image_m


# In[3]:


def batch_norm(inputs, is_training, decay=.5, epsilon=0.00000001):
    with tf.name_scope("batch_norm") as scope:


        scale = tf.get_variable("scale_BN", (inputs.get_shape()[1:4]), initializer=tf.ones_initializer())
        beta = tf.get_variable("beta_BN", (inputs.get_shape()[1:4]), initializer=tf.zeros_initializer())
        pop_mean = tf.get_variable("pop_mean", (inputs.get_shape()[1:4]), initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable("pop_var", (inputs.get_shape()[1:4]), initializer=tf.ones_initializer(), trainable=False)

        mean = tf.cond(tf.cast(is_training,tf.bool), lambda: tf.nn.moments(inputs,[0])[0], lambda: tf.multiply(tf.ones(inputs.get_shape()[1:4]), pop_mean))
        var = tf.cond(tf.cast(is_training,tf.bool), lambda: tf.nn.moments(inputs,[0])[1], lambda: tf.multiply(tf.ones(inputs.get_shape()[-1]), pop_var))
        train_mean = tf.cond(tf.cast(is_training,tf.bool), lambda:tf.assign(pop_mean, pop_mean*decay+mean*(1-decay)),lambda:tf.zeros(1))
        train_var = tf.cond(tf.cast(is_training,tf.bool),lambda:tf.assign(pop_var, pop_var*decay+var*(1-decay)),lambda:tf.zeros(1))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, mean, var, beta, scale, epsilon)


# In[4]:





def partial_conv(pixel, mask,is_training, kernel_size, filter_numbers, stride, batch_n, nonlinearity, trans):

    with tf.name_scope("part_conv") as scope:
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]
        if trans==True:
            kernel_d = filter_numbers
            kernel_o = pixel.get_shape().as_list()[3]
        elif trans==False:
            kernel_d = pixel.get_shape().as_list()[3]
            kernel_o = filter_numbers
        elif trans=="same_pad":
            #kernel_d = pixel.get_shape().as_list()[3]
            #kernel_o = filter_numbers
            kernel_d = filter_numbers
            kernel_o = pixel.get_shape().as_list()[3]
        elif trans=="one":
            kernel_d = pixel.get_shape().as_list()[3]
            kernel_o = filter_numbers
            
            
        W = tf.get_variable('Weights', (kernel_h, kernel_w, kernel_d, kernel_o),
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        #tf.add_to_collection('weights', W)
        #print(W.name)
        W1 = tf.ones((kernel_h, kernel_w, kernel_d, kernel_o), name='Weights_mask')

        Z1 = tf.multiply(pixel, mask, name="element_op")

        if trans==True:
            #need to fix for variable last batch size. The last mini_batch will be of different size most of the time
            out_shape_list = pixel.get_shape().as_list()
            out_shape_list[1] = pixel.get_shape().as_list()[1] + 2
            out_shape_list[2] = pixel.get_shape().as_list()[2] + 2
            out_shape_list[3] = filter_numbers
            out_shape = tf.constant(out_shape_list)
            #out_shape = tf.TensorShape(out_shape_list)
            #out_shape = tf.cast(out_shape,tf.int32)
            prime_conv = tf.nn.conv2d_transpose(Z1, W,out_shape, strides=stride, padding="VALID", name="prime_conv")
            sec_conv = tf.nn.conv2d_transpose(mask, W1,output_shape=tf.TensorShape(out_shape_list), strides=stride, padding="VALID", name="sec_conv")
        elif trans==False:
            prime_conv = tf.nn.conv2d(Z1, W, strides=stride, padding="VALID", name="prime_conv")
            sec_conv = tf.nn.conv2d(mask, W1, strides=stride, padding="VALID", name="sec_conv")
        elif trans=="same_pad":
            #prime_conv = tf.nn.conv2d(Z1, W, strides=stride, padding="SAME", name="prime_conv")
            #sec_conv = tf.nn.conv2d(mask, W1, strides=stride, padding="SAME", name="sec_conv")
            out_shape_list = pixel.get_shape().as_list()
            out_shape_list[1] = pixel.get_shape().as_list()[1] 
            out_shape_list[2] = pixel.get_shape().as_list()[2] 
            out_shape_list[3] = filter_numbers
            out_shape = tf.constant(out_shape_list)
            prime_conv = tf.nn.conv2d_transpose(Z1, W,out_shape, strides=stride, padding="SAME", name="prime_conv")
            sec_conv = tf.nn.conv2d_transpose(mask, W1,output_shape=tf.TensorShape(out_shape_list), strides=stride, padding="SAME", name="sec_conv")
        elif trans=="one":
            prime_conv = tf.nn.conv2d(Z1, W, strides=stride, padding="VALID", name="prime_conv")
            sec_conv = tf.nn.conv2d(mask, W1, strides=stride, padding="VALID", name="sec_conv")
            

        inver_sum = tf.divide(tf.constant(1.0), sec_conv)
        clean_sum = tf.where(tf.is_inf(inver_sum), tf.zeros_like(inver_sum), inver_sum)

        weighted_pixel = tf.multiply(prime_conv, clean_sum, name="multi_inver_sum")
        up_mask = tf.where(tf.not_equal(sec_conv, tf.constant(0.0)),tf.ones_like(sec_conv),sec_conv)

        #normalized_out = tf.cond(tf.cast(batch_n,tf.bool), lambda:batch_norm(weighted_pixel, tf.cast(is_training,tf.bool)), lambda:weighted_pixel)
        #B = tf.get_variable('Biases',(1,weighted_pixel.get_shape()[1],weighted_pixel.get_shape()[2],weighted_pixel.get_shape()[3]),
        #                    initializer=tf.constant_initializer(.01))
        B = tf.get_variable('Biases',(1,1,1,prime_conv.get_shape()[3]),
                            initializer=tf.constant_initializer(.01))
        
        normalized_out = tf.add(weighted_pixel,B)
        #normalized_out = weighted_pixel
        
        if nonlinearity=="relu":
            up_pixel = tf.nn.relu(normalized_out, name="relu")
        elif nonlinearity=="leaky_relu":
            up_pixel = tf.nn.leaky_relu(normalized_out, name="leaky_relu")
        elif nonlinearity=="none":
            #up_pixel = normalized_out
            up_pixel = tf.sigmoid(normalized_out)
            #up_pixel = tf.nn.relu(normalized_out, name="relu")
        elif nonlinearity=="elu":
            up_pixel = tf.keras.activations.elu(normalized_out)
            
        tf.summary.histogram("weights", W)    
        tf.summary.histogram("biases", B)   
        tf.summary.histogram("activations", up_pixel)   
        
        return up_pixel, up_mask
    


def place_holders(mini_size,height, width, channels):
    #X = tf.placeholder(tf.float32, shape=(mini_size, height, width, channels))
    Y = tf.placeholder(tf.float32, shape=(mini_size, height, width, channels))
    M = tf.placeholder(tf.float32, shape=(mini_size, height, width, channels))
    return M ,Y


def near_up_sampling(pixel, mask, output_size):
    with tf.name_scope("nearest_up") as scope:
        up_pixel = tf.image.resize_nearest_neighbor(pixel, size=output_size, name="nearest_pixel_up")
        up_mask = tf.image.resize_nearest_neighbor(pixel, size=output_size, name="nearest_mask_up")
        return up_pixel, up_mask

def concat(near_pixel, pconv_pixel, near_mask, pconv_mask):
    with tf.name_scope("concatenation") as scope:
        up_pixel = tf.concat([pconv_pixel, near_pixel], axis=3)
        up_mask = tf.concat([pconv_mask,near_mask], axis=3)
        return up_pixel, up_mask

def decoding_layer(pixel_in,mask_in,is_training, output_size_in, pconv_pixel1, pconv_mask1, filter_numbers1):
    with tf.name_scope("decoding") as scope:
        near_pixel1,near_mask1 = near_up_sampling(pixel_in,mask_in,output_size_in)
        concat_pixel,concat_mask = concat(near_pixel1, pconv_pixel1, near_mask1, pconv_mask1)
        pixel_out,mask_out = partial_conv(concat_pixel,concat_mask,is_training,[3,3],filter_numbers1,[1,1,1,1],
                                        True,"leaky_relu",trans=True)
        return pixel_out,mask_out


# In[5]:


def forward_prop(is_training, pixel, mask):
    non_lin = "relu"
    
#     with tf.variable_scope("PConv1") as scope:
#         p_out1,m_out1 = partial_conv(pixel,mask,is_training,kernel_size=[3,3],filter_numbers=64,stride=[1,2,2,1],
#                                     batch_n=False,nonlinearity="relu",trans=False)
    with tf.variable_scope("PConv2") as scope:
        p_out2,m_out2 = partial_conv(pixel,mask,is_training,kernel_size=[3,3],filter_numbers=4,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv3") as scope:
        p_out3,m_out3 = partial_conv(p_out2,m_out2,is_training,kernel_size=[3,3],filter_numbers=8,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv4") as scope:
        p_out4,m_out4 = partial_conv(p_out3,m_out3,is_training,kernel_size=[3,3],filter_numbers=16,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv5") as scope:
        p_out5,m_out5 = partial_conv(p_out4,m_out4,is_training,kernel_size=[3,3],filter_numbers=16,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv6") as scope:
        p_out6,m_out6 = partial_conv(p_out5,m_out5,is_training,kernel_size=[3,3],filter_numbers=32,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv7") as scope:
        p_out7,m_out7 = partial_conv(p_out6,m_out6,is_training,kernel_size=[3,3],filter_numbers=32,stride=[1,2,2,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("PConv8") as scope:
        p_out8,m_out8 = partial_conv(p_out7,m_out7,is_training,kernel_size=[3,3],filter_numbers=32,stride=[1,1,1,1],
                                    batch_n=True,nonlinearity=non_lin,trans=False)
    with tf.variable_scope("decoding9") as scope:
        p_out9,m_out9 = decoding_layer(p_out8,m_out8,is_training,(p_out7.get_shape().as_list()[1],p_out7.get_shape().as_list()[2]),
                                        p_out7,m_out7,filter_numbers1=32)

    with tf.variable_scope("decoding10") as scope:
        p_out10,m_out10 = decoding_layer(p_out9,m_out9,is_training,(p_out6.get_shape().as_list()[1],p_out6.get_shape().as_list()[2]),
                                        p_out6,m_out6,filter_numbers1=32)

    with tf.variable_scope("decoding11") as scope:
        p_out11,m_out11 = decoding_layer(p_out10,m_out10,is_training,(p_out5.get_shape().as_list()[1],p_out5.get_shape().as_list()[2]),
                                        p_out5,m_out5,filter_numbers1=16)

    with tf.variable_scope("decoding12") as scope:
        p_out12,m_out12 = decoding_layer(p_out11,m_out11,is_training,(p_out4.get_shape().as_list()[1],p_out4.get_shape().as_list()[2]),
                                        p_out4,m_out4,filter_numbers1=16)

    with tf.variable_scope("decoding13") as scope:
        p_out13,m_out13 = decoding_layer(p_out12,m_out12,is_training,(p_out3.get_shape().as_list()[1],p_out3.get_shape().as_list()[2]),
                                        p_out3,m_out3,filter_numbers1=8)

    with tf.variable_scope("decoding14") as scope:
        p_out14,m_out14 = decoding_layer(p_out13,m_out13,is_training,(p_out2.get_shape().as_list()[1],p_out2.get_shape().as_list()[2]),
                                        p_out2,m_out2,filter_numbers1=4)

#     with tf.variable_scope("decoding15") as scope:
#         p_out15,m_out15 = decoding_layer(p_out14,m_out14,is_training,(p_out1.get_shape().as_list()[1],p_out1.get_shape().as_list()[2]),
#                                         p_out1,m_out1,filter_numbers1=64)

    #with tf.variable_scope("decoding16") as scope:
    #    p_out16,m_out16 = decoding_layer(p_out15,m_out15,is_training,(pixel.get_shape().as_list()[1],pixel.get_shape().as_list()[2]),
    #                                    pixel,mask,filter_numbers1=1)

    
    
    
    with tf.variable_scope("decoding15") as scope:
        near_pixel1,near_mask1 = near_up_sampling(p_out14,m_out14,(pixel.get_shape().as_list()[1],pixel.get_shape().as_list()[2]))
        pixel_hole = tf.multiply(pixel, mask, name="multiply_mask")
        concat_pixel,concat_mask = concat(near_pixel1, pixel_hole, near_mask1, mask)
        pixel_out,mask_out = partial_conv(concat_pixel,concat_mask,is_training,[1,1],filter_numbers=1,stride=[1,1,1,1],
                                        batch_n=False,nonlinearity="none",trans="one")
    
    return pixel_out,mask_out



def compute_cost(pixel_gt,mask_gt,pixel_pre,hole_pera,valid_pera):
    with tf.name_scope("cost") as scope:
        loss_valid = tf.losses.absolute_difference(tf.multiply(pixel_gt,mask_gt),tf.multiply(pixel_pre,mask_gt), weights=1.0,
                                                   reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        loss_hole = tf.losses.absolute_difference(tf.multiply(pixel_gt,(1-mask_gt)),tf.multiply(pixel_pre,(1-mask_gt)), weights=1.0,
                                                    reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        
        #loss_hole = loss_hole * loss_hole
        #loss_valid = loss_valid * loss_valid 
        #loss_valid = tf.losses.mean_squared_error(tf.multiply(pixel_gt,mask_gt),tf.multiply(pixel_pre,mask_gt))
        #loss_hole = tf.losses.mean_squared_error(tf.multiply(pixel_gt,(1-mask_gt)),tf.multiply(pixel_pre,(1-mask_gt)))

        #total_loss = (tf.multiply(valid_pera,loss_valid) + tf.multiply(hole_pera,loss_hole))/(hole_pera+valid_pera)
        #total_loss = (loss_valid + tf.multiply(hole_pera,loss_hole))/(hole_pera)
        total_loss = (valid_pera*loss_valid + hole_pera*loss_hole)/(hole_pera+valid_pera)

        tf.summary.scalar('loss',total_loss)
   
        return total_loss


# In[6]:


def model(learning_rate,num_epochs,mini_size,break_t,break_v,pt_out,hole_pera,valid_pera,decay_s,decay_rate,
         fil_num):
    #ops.reset_default_graph()
    tf.summary.scalar('learning_rate',learning_rate)
    tf.summary.scalar('batch_size',mini_size)
    tf.summary.scalar('training_break',break_t)
    tf.summary.scalar('validation_break',break_v)
    tf.summary.scalar('print_interval',pt_out)
    tf.summary.scalar('hole_loss_weight',hole_pera)
    tf.summary.scalar('valid_loss_weight',valid_pera)
    tf.summary.scalar('decay_steps',decay_s)
    tf.summary.scalar('decay_rate',decay_rate)
    tf.summary.scalar('max_filter_number',fil_num)

    m = 19488
    #m = 8
    #h = 512
    #w = 512
    #c = 1
    
    m_val_size = 1888
        
    #filenames = "/media/antor/Files/ML/Papers/train_mfix.tfrecords"
    filenames = tf.placeholder(tf.string)
    is_training = tf.placeholder(tf.bool)
    
    
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(num_epochs)
    #dataset = dataset.shuffle(5000)
    dataset = dataset.batch(mini_size)
    iterator = dataset.make_initializable_iterator(shared_name="iter")
    #tf.add_to_collection('iterator', iterator)
    pix_gt, mask_in = iterator.get_next()
    
    pix_gt = tf.reshape(pix_gt,[mini_size,256,256,1])
    mask_in = tf.reshape(mask_in,[mini_size,256,256,1])
    
    tf.summary.image("input_Y",pix_gt,3)
    tf.summary.image("input_M",mask_in,3)
    
    pixel_out, mask_out = forward_prop(is_training=is_training,pixel=pix_gt, mask=mask_in)
    
    
    tf.summary.image("output_Y",pixel_out,3)
    tf.summary.image("output_M",mask_out,3)
    
    cost = compute_cost(pixel_gt=pix_gt, mask_gt=mask_in, pixel_pre=pixel_out, hole_pera=hole_pera,valid_pera=valid_pera)
    
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate_d = tf.train.exponential_decay(learning_rate, global_step,decay_s,decay_rate, staircase=False)
    #tf.summary.scalar('learning_rate_de',learning_rate_d)
    
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d,name="adam").minimize(cost,global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name="adam").minimize(cost)
    
    num_mini = int(m/mini_size)          #must keep this fully divided and num_mini output as int pretty sure it doesn't need
                                    #to be an int    
    merge_sum = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())   #for tensorboard
    
    saver = tf.train.Saver()    #for model saving
    #builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')

    init = tf.global_variables_initializer()
    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    
    #sess.run(init)
    saver.restore(sess,('/media/antor/Files/main_projects/gitlab/Landsat7_image_inpainting/tf_models/run-20181010102038/my_model.ckpt'))
    
    sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/Papers/test.tfrecords"})
    
    mini_cost = 0.0
    counter = 1
    epoch_cost = 0.0
    epoch = 0
    
    path = "/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/mini_gt/*.jpg"

    only_need = glob.glob(path)
    #only_need = only_need[0:1]

    onlyfiles = [os.path.basename(x) for x in only_need]

    onlyn = [os.path.splitext(x)[0] for x in onlyfiles]
    
    num = 0
    while True:
        try:
            
            #_ , temp_cost = sess.run([optimizer,cost], feed_dict={is_training:True})
            output = sess.run(pixel_out, feed_dict={is_training:True})
            last_y = output*255
            #for i in range(last_y.shape[0]):
            #      cv2.imwrite("/media/antor/Files/ML/fixed/plane"+str(i)+".png", last_y[i])
            
            cv2.imwrite("/media/antor/Files/ML/Papers/river_bank/JPEG_(copy)/inpain_test/fixed/"+str(onlyn[num])+".jpg", last_y[0])
            print(num)
            num+=1
        except tf.errors.OutOfRangeError:
            break
    sess.close()


# In[7]:


model(learning_rate=.00960955,num_epochs=1,mini_size=1,break_t=7000,break_v=700,pt_out=20,hole_pera=6.0,
      valid_pera=1.0,decay_s=538.3,decay_rate=.96,fil_num=32)

