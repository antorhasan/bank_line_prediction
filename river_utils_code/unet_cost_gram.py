
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

a = tf.random_uniform(shape=(1,508,508,1),minval=0,maxval=1)

ma = tf.nn.max_pool(a,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID",data_format='NHWC',name=None)

sess= tf.Session
#y = sess.run(ma)
print(ma.shape)
sess.close()

def _parse_function(example_proto):
    features = {
                "image_y": tf.FixedLenFeature((), tf.string ),
                "image_m": tf.FixedLenFeature((), tf.string ),
                "image_x": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    
    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)
    image_m = tf.decode_raw(parsed_features["image_m"],  tf.float64)
    image_x = tf.decode_raw(parsed_features["image_x"],  tf.float64)
    
    image_y = tf.cast(image_y,dtype=tf.float32)
    image_m = tf.cast(image_m,dtype=tf.float32)
    image_x = tf.cast(image_x,dtype=tf.float32)
    
    image_y = tf.reshape(image_y, [512,512,1])
    image_m = tf.reshape(image_m, [512,512,1])
    image_x = tf.reshape(image_x, [512,512,1])
    
    return image_y,image_m,image_x

def batch_norm(inputs, is_training, decay=.99, epsilon=0.00000001):
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


def place_holders(mini_size,height, width, channels):
    X = tf.placeholder(tf.float32, shape=(mini_size, height, width, channels))
    
    return X

def conv_block(inputs,is_training,kernel_size,filter_numbers,stride,pad,batch_n,nonlinearity):
    with tf.name_scope("conv") as scope:
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]
        kernel_d = pixel.get_shape().as_list()[3]
        kernel_o = filter_numbers
        
        W = tf.get_variable('Weights', (kernel_h, kernel_w, kernel_d, kernel_o),
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        if pad=="VALID":
            prime_conv = tf.nn.conv2d(inputs, W, strides=stride, padding="VALID", name="prime_conv")
        if pad=="SAME":
            prime_conv = tf.nn.conv2d(inputs, W, strides=stride, padding="SAME", name="prime_conv")
        
        normalized_out = tf.cond(tf.cast(batch_n,tf.bool),lambda:batch_norm(weighted_pixel, is_training),lambda:weighted_pixel)
        
        if nonlinearity=="relu":
            up_pixel = tf.nn.relu(normalized_out, name="relu")
        elif nonlinearity=="leaky_relu":
            up_pixel = tf.nn.leaky_relu(normalized_out, name="leaky_relu")
        elif nonlinearity=="none":
            up_pixel = normalized_out
            
        return up_pixel
    
def near_up_sampling(pixel_up,pixel_do,output_size):
    with tf.name_scope("up_concat") as scope:
        up_pixel = tf.image.resize_nearest_neighbor(pixel_up, size=output_size, name="nearest_pixel_up")
        last_pixel = tf.concat([pixel_do,up_pixel], axis=3)
        return last_pixel

    
    

def forward_prop(is_training, pixel):
    with tf.variable_scope("Conv1") as scope:
        p_out1 = conv_block(pixel,is_training,[3,3],filter_numbers=64,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Conv2") as scope:
        p_out2 = conv_block(p_out1,is_training,[3,3],filter_numbers=64,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Pool1") as scope:
        p_out3 = tf.nn.max_pool(p_out2,ksize=[2,2],strides=[1,2,2,1],padding="VALID",data_format='NHWC',name=None)
        
    with tf.variable_scope("Conv3") as scope:
        p_out4 = conv_block(p_out3,is_training,[3,3],filter_numbers=128,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Conv4") as scope:
        p_out5 = conv_block(p_out4,is_training,[3,3],filter_numbers=128,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
    
    with tf.variable_scope("Pool2") as scope:
        p_out6 = tf.nn.max_pool(p_out5,ksize=[2,2],strides=[1,2,2,1],padding="VALID",data_format='NHWC',name=None)
        
    with tf.variable_scope("Conv5") as scope:
        p_out7 = conv_block(p_out6,is_training,[3,3],filter_numbers=256,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Conv6") as scope:
        p_out8 = conv_block(p_out7,is_training,[3,3],filter_numbers=256,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Pool3") as scope:
        p_out9 = tf.nn.max_pool(p_out8,ksize=[2,2],strides=[1,2,2,1],padding="VALID",data_format='NHWC',name=None)
        
    with tf.variable_scope("Conv7") as scope:
        p_out10 = conv_block(p_out9,is_training,[3,3],filter_numbers=512,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
    
    with tf.variable_scope("Conv8") as scope:
        p_out11 = conv_block(p_out10,is_training,[3,3],filter_numbers=512,stride=[1,1,1,1],pad="VALID",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Unpool1") as scope:
        p_out12 = near_up_sampling(p_out11,p_out8,(p_out8.get_shape().as_list()[1],p_out8.get_shape().as_list()[2]))
        
    with tf.variable_scope("Conv9") as scope:
        p_out13 = conv_block(p_out12,is_training,[3,3],filter_numbers=256,stride=[1,1,1,1],pad="SAME",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Conv10") as scope:
        p_out14 = conv_block(p_out13,is_training,[3,3],filter_numbers=256,stride=[1,1,1,1],pad="SAME"batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Unpool2") as scope:
        p_out15 = near_up_sampling(p_out14,p_out5,(p_out5.get_shape().as_list()[1],p_out5.get_shape().as_list()[2]))
    
    with tf.variable_scope("Conv11") as scope:
        p_out16 = conv_block(p_out15,is_training,[3,3],filter_numbers=128,stride=[1,1,1,1],pad="SAME",batch_n=True,nonlinearity="relu")
    
    with tf.variable_scope("Conv12") as scope:
        p_out17 = conv_block(p_out16,is_training,[3,3],filter_numbers=128,stride=[1,1,1,1],pad="SAME",batch_n=True,nonlinearity="relu")
        
    with tf.variable_scope("Unpool3") as scope:
        p_out18 = near_up_sampling(p_out17,p_out2,(p_out2.get_shape().as_list()[1],p_out2.get_shape().as_list()[2]))
    
    with tf.variable_scope("Conv13") as scope:
        p_out19 = conv_block(p_out18,is_training,[3,3],filter_numbers=64,stride=[1,1,1,1],pad="SAME",batch_n=True,nonlinearity="relu")
    
    with tf.variable_scope("Conv14") as scope:
        p_out20 = conv_block(p_out19,is_training,[3,3],filter_numbers=64,stride=[1,1,1,1],pad="SAME",batch_n=True,nonlinearity="relu")
    
    with tf.variable_scope("Conv15") as scope:
        p_out21 = conv_block(p_out20,is_training,[3,3],filter_numbers=1,stride=[1,1,1,1],pad="SAME",batch_n=True,nonlinearity="relu")
        
return p_out21       
        
        
        
        
def compute_cost(labels,predictions):
    
    total_loss =  tf.losses.mean_squared_error(labels,predictions,weights=1.0)
    return total_loss
               
        

def model(learning_rate,num_epochs,mini_size):
    #ops.reset_default_graph()
#     m = x_train.shape[0]
#     h = x_train.shape[1]
#     w = x_train.shape[2]
#     c = x_train.shape[3]
    m = 9882
    h = 512
    w = 512
    c = 1
    
    m_val_size = 1098
    costs = []

    X = place_holders(mini_size,h, w, c)
    is_training = tf.placeholder(tf.bool,name="training")
    
    pixel_out = forward_prop(is_training, pixel=X)

    cost = compute_cost(X,pixel_out)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    
    #data_x = tf.placeholder(x_train.dtype, x_train.shape)
    #data_m = tf.placeholder(m_train.dtype, m_train.shape)
    #data_y = tf.placeholder(y_train.dtype, y_train.shape)
    
    
    
    filenames = "F:/D/Papers/train.tfrecords"
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    #dataset = tf.data.Dataset.from_tensor_slices((data_x, data_m, data_y))
    
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(100)
    dataset = dataset.batch(mini_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    num_mini = int(m/mini_size)          #must keep this fully divided and num_mini output as int pretty sure it doesn't need
                                    #to be an int
    
    l1_summary = tf.summary.scalar('L1', cost)            #for tensorboard
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())   #for tensorboard
    
    saver = tf.train.Saver()
    
    sess = tf.Session()
    sess.run(init)
    #sess.run(iterator.initializer, feed_dict={data_x:x_train, data_m:m_train, data_y:y_train})
    sess.run(iterator.initializer)
    mini_cost = 0.0
    counter = 1
    while True:
        try:
            pix_in, mask_in, label_in= sess.run(next_element)
            pix_in = tf.reshape(pix_in,[mini_size,512,512,1])
            mask_in = tf.reshape(mask_in,[mini_size,512,512,1])
            label_in = tf.reshape(label_in,[mini_size,512,512,1])
            pix_in = pix_in.eval(session=sess)
            mask_in = mask_in.eval(session=sess)
            label_in = label_in.eval(session=sess)
            
            
            _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:pix_in, M:mask_in, Y:label_in, is_training:True})
            
            mini_cost += temp_cost/num_mini
            if counter%num_mini==0:
                print("cost after epoch " + str(counter/num_mini) + ": " + str(mini_cost))
                mini_cost =0.0 
            
            if counter%30==0:         #for tensorboard
                summary_str = l1_summary.eval(feed_dict={X:pix_in, M:mask_in, Y:label_in, is_training:True})
                #step = int(counter%num_mini) * n_batches + batch_index
                file_writer.add_summary(summary_str, counter)
                
            counter = counter + 1
        except tf.errors.OutOfRangeError:
            #counter = counter + 1
            #print("cost after epoch " + str(counter) + ": " + str(mini_cost))
            #mini_cost = 0.0
            break
    
    file_writer.close()            #for tensorboard
    #data_x_val = tf.placeholder(x_val.dtype, x_val.shape)
    #m_val = m_val.astype(float)
    #data_m_val = tf.placeholder(m_val.dtype, m_val.shape)
    #data_y_val = tf.placeholder(y_val.dtype, y_val.shape)
    
    filenames_val = "F:/D/Papers/val.tfrecords"
    dataset_val = tf.data.TFRecordDataset(filenames_val)
    dataset_val = dataset_val.map(_parse_function)
    
    #dataset_val = tf.data.Dataset.from_tensor_slices((data_x_val,data_m_val,data_y_val))
    dataset_val = dataset_val.shuffle(100)
    dataset_val = dataset_val.batch(mini_size)
    
    iterator_val = dataset_val.make_initializable_iterator()
    next_element_val = iterator_val.get_next()
    num_mini_val = int(m_val_size/mini_size)

    counter_val = 1
    #sess.run(iterator_val.initializer, feed_dict={data_x_val:x_val,data_m_val:m_val,data_y_val:y_val})
    sess.run(iterator_val.initializer)
    mini_cost_val = 0.0
    while True:
        try:
            pix_in_val, mask_in_val, label_in_val= sess.run(next_element_val)
            pix_in_val = tf.reshape(pix_in_val,[mini_size,512,512,1])
            mask_in_val = tf.reshape(mask_in_val,[mask_in_val,512,512,1])
            label_in_val = tf.reshape(label_in_val,[label_in_val,512,512,1])
            
            temp_cost_val = sess.run(cost, feed_dict={X:pix_in_val, M:mask_in_val,Y:label_in_val,is_training:False})
            mini_cost_val += temp_cost_val/num_mini_val
            if counter%num_mini_val==0:
                print("cost after epoch " + str(counter_val/num_mini_val) + ": " + str(mini_cost_val))
                mini_cost_val =0.0 
            counter_val = counter_val + 1
        except tf.errors.OutOfRangeError:

            #print("validation cost after epoch " + "1" + ": " + str(mini_cost_val))

            break
    
    save_path = saver.save(sess, "F:/D/Papers/my_model_final.ckpt")
    sess.close()

