
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp


# In[ ]:




sess = tf.Session()
graph = tf.get_default_graph()



with graph.as_default():
    with sess.as_default():
        is_training = tf.placeholder(tf.bool)
        filenames = tf.placeholder(tf.string)
        #restoring the model
        saver = tf.train.import_meta_graph('/home/antor/Documents/gitlab/Landsat7_image_inpainting/tf_models/run-20181005194313/my_model.ckpt.meta')
        #chkp.print_tensors_in_checkpoint_file("/home/antor/Documents/gitlab/Landsat7_image_inpainting/tf_models/last_saver/my_model.ckpt", tensor_name='', all_tensors=True)
        saver.restore(sess,('/home/antor/Documents/gitlab/Landsat7_image_inpainting/tf_models/run-20181005194313/my_model.ckpt'))
#         dataset = tf.data.TFRecordDataset(filenames)
#         dataset = dataset.map(_parse_function)
#         dataset = dataset.repeat(1)
#         dataset = dataset.shuffle(200)
#         dataset = dataset.batch(16)
#         iterator = dataset.make_initializable_iterator()
        
        #iterator = tf.get_collection('iterator')[0]
        #sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/Papers/train_last.tfrecords"})
        #train_op = tf.get_collection('train_op')[0]
        #cost_op = tf.get_collection('cst')[0]
        
        
        
        #print(sess.run("PConv2/Weights:0"))
        #print (sess.run(tf.get_default_graph().get_tensor_by_name('w1:0')))
        print(tf.global_variables())
        for op in tf.get_default_graph().get_operations():
            print(str(op.name))
        #model(learning_rate=.00960955,num_epochs=1,mini_size=16,break_t=7000,break_v=200,pt_out=20,hole_pera=6.0,valid_pera=1.0)
        #sess.run(tf.global_variables_initializer())
        #x_tensor = graph.get_tensor_by_name("PConv2/Biases:0")
        #train_op = tf.get_collection('train_op')
        #cost_op = tf.get_collection('cst')
        #iter_op = tf.get_collection('iterator')
        l_r = tf.get_default_graph().get_operation_by_name("learning_rate")
        l_r_1 = tf.get_default_graph().get_operation_by_name("learning_rate/values")
        print(sess.run(l_r))
        print(sess.run(l_r_1))
        
        op_iter = tf.get_default_graph().get_operation_by_name("IteratorGetNext")
        #op_iter = tf.get_default_graph().get_operation_by_name("Iterator")
        #op_iter = tf.get_default_graph().get_operation_by_name("IteratorToStringHandle")
        #op_iter = tf.get_default_graph().get_operation_by_name("MakeIterator")
        cost_op = tf.get_default_graph().get_operation_by_name("cost/loss")
        train_op = tf.get_default_graph().get_operation_by_name("adam")
        
        sess.run(op_iter.initialize,feed_dict={filenames:'/media/antor/Files/ML/Papers/train_last.tfrecords'})
        for step in range(10):
            _ , temp_cost = sess.run([train_op,cost_op], feed_dict={is_training:True})
            print(temp_cost)
        #print(sess.run(x_tensor))


# In[ ]:


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
    
    image_y = tf.cast(image_y,dtype=tf.float32)
    image_m = tf.cast(image_m,dtype=tf.float32)

    return image_y,image_m


# In[ ]:


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
    dataset = dataset.shuffle(200)
    dataset = dataset.batch(mini_size)
    iterator = dataset.make_initializable_iterator()
 
    pix_gt, mask_in = iterator.get_next()
    
    pix_gt = tf.reshape(pix_gt,[mini_size,256,256,1])
    mask_in = tf.reshape(mask_in,[mini_size,256,256,1])


# In[ ]:


sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/Papers/train_last.tfrecords"})

