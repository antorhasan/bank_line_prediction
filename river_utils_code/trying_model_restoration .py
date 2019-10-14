
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
#dir = os.path.dirname(os.path.realpath(__file__))


# In[ ]:


with tf.Session() as sess:  
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], '/home/antor/Documents/gitlab/Landsat7_image_inpainting/tf_models/run-20181005142626/')
    w1 = sess.run("PConv2/Weights:0")
    #print(w1)


# In[ ]:


#init = tf.global_variables_initializer()
with tf.device('/gpu:0'):
    saver = tf.train.import_meta_graph('/home/antor/Documents/gitlab/Landsat7_image_inpainting/tf_models/run-20181004213154/my_model_fin.meta')
    graph = tf.get_default_graph()
    global_step_tensor = graph.get_tensor_by_name('PConv2/Weights:0')

    with tf.Session() as sess:
        # To initialize values with saved data
        saver.restore(sess, "/home/antor/Documents/gitlab/Landsat7_image_inpainting/tf_models/run-20181004213154/./")
        print(sess.run(global_step_tensor)) # returns 1000

