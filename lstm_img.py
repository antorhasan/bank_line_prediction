import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, ConvLSTM2D, Reshape, LSTM, Flatten
#import cv2
from datetime import datetime

tf.enable_eager_execution()


def _parse_function_img(example_proto):

    features = {
            "image": tf.io.FixedLenFeature((), tf.string),
            "msk": tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    image = tf.io.decode_raw(parsed_features["image"],  tf.float32)
    msk = tf.io.decode_raw(parsed_features["msk"],  tf.float32)

    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image,[256,256,3])
    msk = tf.cast(msk, dtype=tf.float32)
    return image


def _parse_function_msk(example_proto):

    features = {
            "image": tf.io.FixedLenFeature((), tf.string),
            "msk": tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    #image = tf.io.decode_raw(parsed_features["image"],  tf.float32)
    msk = tf.io.decode_raw(parsed_features["msk"],  tf.float32)

    msk = tf.cast(msk, dtype=tf.float32)
    msk = tf.reshape(msk,[256,])

    return msk

class Conv_layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Conv_layer, self).__init__()
        self.conv1 = Conv2D(32,(3,3),padding='valid',strides=(1,1),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max1 = MaxPool2D((2,2))
        self.conv2 = Conv2D(64,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max2 = MaxPool2D((2,2))
        self.conv3 = Conv2D(64,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max3 = MaxPool2D((2,2))
        self.conv4 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max4 = MaxPool2D((2,2))
        self.conv5 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max5 = MaxPool2D((2,2))
        self.conv6 = Conv2D(256,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max6 = MaxPool2D((2,2))
        #self.reshape1 = Reshape((1,256,256,3))
        #self.reshape2 = Reshape((3,3,256))
        #self.conv7 = Conv2D(256,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self, inputs):
        #x = self.reshape1(inputs)
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.max4(x)
        x = self.conv5(x)
        x = self.max5(x)
        x = self.conv6(x)
        x = self.max6(x)
        #x = self.conv7(x)
        #o = self.reshape2(x)
        return x

    def model(self):
        #x = np.zeros((1,256,256,3), dtype=np.float32)
        #x = tf.random.uniform((1,256,256,3))
        x = tf.keras.layers.Input(shape=(256,256,3), dtype=tf.float32)
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()

#conv = Conv_layer()
#conv.model()

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layer = Conv_layer()
        #self.reshape = Reshape((1,2,3,3,256))
        self.convlstm = ConvLSTM2D(256,(3,3), data_format='channels_last',padding='same',return_sequences=False)
        #self.lstm = LSTM(256,bias_initializer=tf.keras.initializers.constant(.01),activation='relu')
        #self.flat = Flatten()
        #self.reshape_d = Reshape((2,2,256))  #first dimension is batch size,second is 2 input images
        self.dense1 = Dense(256, activation='tanh',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')
        self.reshape_output = Reshape((2,-1))   #it's running due to this.something is very off with the reshape
        self.reshape_last = Reshape((2,1,256))
        self.res = Reshape((2,-1,3,256))
        self.conv7 = Conv2D(256,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.map_fn(self.conv_layer, inputs) # input needs to be of shpae (sample,2,256,256,3)
        #x = self.conv_layer(inputs)
        #x = self.reshape(x)
        #x = self.flat(x)
        #x = self.flat(x)
        x = self.convlstm(x)
        #x = self.reshape_d(x)

        #print(x)
        #x = self.res(x)
        x = self.conv7(x)
        #x = self.lstm(x)
        #print(x)
        #print(x)
        x = self.reshape_output(x)
        x = self.dense1(x)
        x = self.reshape_last(x)

        return x
        
    def model(self):
        x = tf.keras.layers.Input(shape=(2,256,256,3), dtype=tf.float32)
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()

#model = MyModel()
#model.model()

def loss_object(labels, predictions):
    loss = tf.keras.losses.mse(labels, predictions)
    return loss

#@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(labels, predictions)
    #train_accuracy(labels, predictions)

#@tf.function
def test_step(images, labels):
	predictions = model(images)
	t_loss = loss_object(labels, predictions)

	test_loss(labels, predictions)
	#test_accuracy(labels, predictions)


def predict_step(images):
    result = model(images)

    lef_mean = np.load('./data/numpy_arrays/left/mean.npy')
    lef_std = np.load('./data/numpy_arrays/left/std.npy')
    lef_a = np.load('./data/numpy_arrays/left/a.npy')
    lef_b = np.load('./data/numpy_arrays/left/b.npy')
    rg_mean = np.load('./data/numpy_arrays/right/mean.npy')
    rg_std = np.load('./data/numpy_arrays/right/std.npy')
    rg_a = np.load('./data/numpy_arrays/right/a.npy')
    rg_b = np.load('./data/numpy_arrays/right/b.npy')
    print(result.shape)
    #coun = 0
    result = np.asarray(result)
    '''this part rescales and renormalizes output according to their
    respective left and right means and stds(the following is for 2 stepsize'''
    for j in range(5404):
        #print(coun)
        if j%4==0 or j%4==1:
            stuff = (result[j] - lef_b) / lef_a
            result[j] = (stuff*lef_std) + lef_mean
        
        if (j-2)%4 == 0 or (j-2)%4 == 1:
            stuff = (result[j] - rg_b) / rg_a
            result[j] = (stuff*rg_std) + rg_mean

    """ for j in range(5404):
        #print(coun)
        if j%2==0 :
            stuff = (result[j] - lef_b) / lef_a
            result[j] = (stuff*lef_std) + lef_mean
        
        if j%2 !=0 :
            stuff = (result[j] - rg_b) / rg_a
            result[j] = (stuff*rg_std) + rg_mean """

        #print(stuff.shape)
        #print(stuff)
    """ print(result[0:30])
    result = (result - b) / a 
    result = (result * std) + mean
    print(result) """
    img1 = np.zeros((1351,1119))
    img2 = np.zeros((1351,1119))
    
    img_1 = cv2.imread('./data/exp1/201601.png',1)
    img_2 = cv2.imread('./data/exp1/201701.png',1)    

    for i in range(int(len(result)/4)):
        for j in range(4):
            img1[i,int(result[4*i+0])] = 255
            img2[i,int(result[4*i+1])] = 255
            img1[i,int(result[4*i+2])] = 255
            img2[i,int(result[4*i+3])] = 255

            img_1[i,int(result[4*i+0])] = [0,0,255]
            img_2[i,int(result[4*i+1])] = [0,0,255]
            img_1[i,int(result[4*i+2])] = [0,0,255]
            img_2[i,int(result[4*i+3])] = [0,0,255]


    #cv2.imwrite('./data/resul_line/'+'year0'+'.png',img1)
    #cv2.imwrite('./data/resul_line/'+'year1'+'.png',img2)

    cv2.imwrite('./data/resul_line/'+'label0'+'.png',img_1)
    cv2.imwrite('./data/resul_line/'+'label1'+'.png',img_2)


dataseti = tf.data.TFRecordDataset('./data/img/record/first_img/train.tfrecords')
dataseti = dataseti.map(_parse_function_img)
#dataset = dataset.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
dataseti = dataseti.window(size=30, shift=30, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(30))
dataseti = dataseti.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataseti = dataseti.flat_map(lambda x: x.window(size=3, shift=3, stride=1,drop_remainder=True))
dataseti = dataseti.flat_map(lambda x: x.batch(3))
#dataset = dataset.shuffle(3000)
dataseti = dataseti.batch(2)

datasetm = tf.data.TFRecordDataset('./data/img/record/first_img/train.tfrecords')
datasetm = datasetm.map(_parse_function_msk)
#dataset = dataset.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
datasetm = datasetm.window(size=30, shift=30, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(30))
datasetm = datasetm.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
datasetm = datasetm.flat_map(lambda x: x.window(size=3, shift=3, stride=1,drop_remainder=True))
datasetm = datasetm.flat_map(lambda x: x.batch(3))
#dataset = dataset.shuffle(3000)
datasetm = datasetm.batch(2)


dataset_vali = tf.data.TFRecordDataset('./data/img/record/first_img/val.tfrecords')
dataset_vali = dataset_vali.map(_parse_function_img)
#dataset_val = dataset_val.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
dataset_vali = dataset_vali.window(size=4, shift=4, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(4))
dataset_vali = dataset_vali.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset_vali = dataset_vali.flat_map(lambda x: x.window(size=3, shift=1, stride=1,drop_remainder=True))
dataset_vali = dataset_vali.flat_map(lambda x: x.batch(3))
#dataset = dataset.shuffle(3000)
dataset_vali = dataset_vali.batch(2)


dataset_valm = tf.data.TFRecordDataset('./data/img/record/first_img/val.tfrecords')
dataset_valm = dataset_valm.map(_parse_function_msk)
#dataset_val = dataset_val.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
dataset_valm = dataset_valm.window(size=4, shift=4, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(4))
dataset_valm = dataset_valm.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset_valm = dataset_valm.flat_map(lambda x: x.window(size=3, shift=1, stride=1,drop_remainder=True))
dataset_valm = dataset_valm.flat_map(lambda x: x.batch(3))
#dataset = dataset.shuffle(3000)
dataset_valm = dataset_valm.batch(2)

model = MyModel()

#model.model()

optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)


train_loss = tf.keras.metrics.MeanSquaredError()

test_loss = tf.keras.metrics.MeanSquaredError()

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#logdir = "./data/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_grads=True, write_images=True)
#callback = [tensorboard_callback]
#tensorboard_callback.set_model(model)

#model.evaluate(callbacks=callback)

EPOCHS = 30
for epoch in range(EPOCHS):
    #for img, msk in zip(dataseti, datasetm):
    #    print(img[:,0:2,:,:],msk[:,2:3,:])
    
    for img, msk in zip(dataseti, datasetm):
        train_step(img[:,0:2,:,:],msk[:,2:3,:])

    for img, msk in zip(dataset_vali, dataset_valm):
        test_step(img[:,0:2,:,:],msk[:,2:3,:])
    
    template = 'Epoch {}, Loss: {}, Test Loss: {},'
    print(template.format(epoch+1,
                        train_loss.result(), test_loss.result() ))

    #tensorboard_callback.on_epoch_end(epoch, logs={'train_loss':train_loss,'test_loss':test_loss})
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

	#train_accuracy.reset_states()

#for data in test:
#    predict_step(data[:, 0:27, :])
