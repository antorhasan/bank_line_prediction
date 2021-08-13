from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, ConvLSTM2D, Reshape, LSTM, Flatten
import cv2
from datetime import datetime

#tf.enable_eager_execution()


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
        self.conv1 = Conv2D(32,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv11 = Conv2D(32,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max1 = MaxPool2D((2,2))
        self.conv2 = Conv2D(64,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv22 = Conv2D(64,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')        
        self.max2 = MaxPool2D((2,2))
        self.conv3 = Conv2D(64,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv33 = Conv2D(64,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max3 = MaxPool2D((2,2))
        self.conv4 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv44 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max4 = MaxPool2D((2,2))
        self.conv5 = Conv2D(256,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv55 = Conv2D(256,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max5 = MaxPool2D((2,2))
        self.conv6 = Conv2D(512,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv66 = Conv2D(512,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max6 = MaxPool2D((2,2))
        #self.reshape1 = Reshape((1,256,256,3))
        #self.reshape2 = Reshape((3,3,256))
        #self.conv7 = Conv2D(256,(3,3),padding='valid',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self, inputs):
        #x = self.reshape1(inputs)
        x = self.conv1(inputs)
        x = self.conv11(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.conv33(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.conv44(x)
        x = self.max4(x)
        x = self.conv5(x)
        x = self.conv55(x)
        x = self.max5(x)
        x = self.conv6(x)
        x = self.conv66(x)
        x = self.max6(x)
        #print(x)
        #print(asd)
        #x = self.conv7(x)
        #o = self.reshape2(x)
        return x


#conv = Conv_layer()
#conv.model()

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_layer = Conv_layer()
        self.convlstm = ConvLSTM2D(512,(3,3), data_format='channels_last',padding='valid',return_sequences=False)
        self.dense1 = Dense(256, activation='tanh',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')

    def call(self, inputs):
        x = tf.map_fn(self.conv_layer, inputs) # input needs to be of shpae (sample,2,256,256,3)
        x = self.convlstm(x)
        x = tf.reshape(x, [2,512])
        x = self.dense1(x)
        #print(x)
        return x

#model = MyModel()
#model.model()

def loss_object(labels, predictions):
    labels = tf.reshape(labels, [2,256])
    predictions = tf.reshape(predictions, [2,256])
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
    #t_loss = loss_object(labels, predictions)
    test_loss(labels, predictions)
    #test_accuracy(labels, predictions)

time_step = 28

dataseti = tf.data.TFRecordDataset('./data/img/record/first_img/train_28.tfrecords')
dataseti = dataseti.map(_parse_function_img)
#dataset = dataset.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
dataseti = dataseti.window(size=28, shift=28, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(28))
dataseti = dataseti.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataseti = dataseti.flat_map(lambda x: x.window(size=time_step, shift=time_step, stride=1,drop_remainder=True))
dataseti = dataseti.flat_map(lambda x: x.batch(time_step))
#dataset = dataset.shuffle(3000)
dataseti = dataseti.batch(2)

datasetm = tf.data.TFRecordDataset('./data/img/record/first_img/train_28.tfrecords')
datasetm = datasetm.map(_parse_function_msk)
#dataset = dataset.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
datasetm = datasetm.window(size=28, shift=28, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(28))
datasetm = datasetm.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
datasetm = datasetm.flat_map(lambda x: x.window(size=time_step, shift=time_step, stride=1,drop_remainder=True))
datasetm = datasetm.flat_map(lambda x: x.batch(time_step))
#dataset = dataset.shuffle(3000)
datasetm = datasetm.batch(2)

dataset = tf.data.Dataset.zip((dataseti, datasetm))
dataset = dataset.shuffle(600)


dataset_vali = tf.data.TFRecordDataset('./data/img/record/first_img/val_28.tfrecords')
dataset_vali = dataset_vali.map(_parse_function_img)
#dataset_val = dataset_val.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
dataset_vali = dataset_vali.window(size=29, shift=29, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(29))
dataset_vali = dataset_vali.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset_vali = dataset_vali.flat_map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
dataset_vali = dataset_vali.flat_map(lambda x: x.batch(time_step))
#dataset = dataset.shuffle(3000)
dataset_vali = dataset_vali.batch(2)


dataset_valm = tf.data.TFRecordDataset('./data/img/record/first_img/val_28.tfrecords')
dataset_valm = dataset_valm.map(_parse_function_msk)
#dataset_val = dataset_val.window(size=2, shift=2, stride=1, drop_remainder=False).flat_map(lambda x: x.batch(2))
dataset_valm = dataset_valm.window(size=29, shift=29, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(29))
dataset_valm = dataset_valm.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset_valm = dataset_valm.flat_map(lambda x: x.window(size=time_step, shift=1, stride=1,drop_remainder=True))
dataset_valm = dataset_valm.flat_map(lambda x: x.batch(time_step))
#dataset = dataset.shuffle(3000)
dataset_valm = dataset_valm.batch(2)

dataset_val = tf.data.Dataset.zip((dataset_vali, dataset_valm))
#dataset_val = dataset_val.shuffle(600)


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

for img, msk in dataset:
    train_step(img[:,0:27,:,:,:],msk[:,27,:])
    model.load_weights('./data/img/model/weights_final_tanh.h5')
    break

array_path = './data/img/numpy_arrays/first_mask/'
mean = np.load(array_path + 'mean.npy')
std = np.load(array_path + 'std.npy')
a = np.load(array_path + 'a.npy')
b = np.load(array_path + 'b.npy')

EPOCHS = 150
for epoch in range(EPOCHS):
    #for img, msk in zip(dataseti, datasetm):
    for img, msk in dataset:
        train_step(img[:,0:27,:,:,:],msk[:,27,:])
    
    #for imgv, mskv in zip(dataset_vali, dataset_valm):
    for imgv, mskv in dataset_val:
        test_step(imgv[:,0:27,:,:,:],mskv[:,27,:])
    #print('k')
    template = 'Epoch {}, Loss: {}, Test Loss: {},'
    print(template.format(epoch+1,
                        train_loss.result(), test_loss.result() ))

    #tensorboard_callback.on_epoch_end(epoch, logs={'train_loss':train_loss,'test_loss':test_loss})
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

	#train_accuracy.reset_states()
    if epoch % 20 == 0 :
        model.save_weights('./data/img/model/weights_final_tanh.h5')

model.save_weights('./data/img/model/weights_final_tanh.h5')

line_1 = []
line_2 = []

left_coor = [679, 700, 652, 601, 582, 508, 452, 440]
right_coor = [1034, 1011, 1010, 1027, 969, 925, 935, 903]

coun = 0
for img, msk in dataset_val:
    #print(coun)
    result = model(img[:,0:27,:,:,:])
    msk = msk[:,27,:]
    #print(result)
    #print(asd)
    result = tf.reshape(result,[2,256])
    msk = tf.reshape(msk,[2,256])
    array_path = './data/img/numpy_arrays/first_mask/'
    mean = np.load(array_path + 'mean.npy')
    std = np.load(array_path + 'std.npy')
    a = np.load(array_path + 'a.npy')
    b = np.load(array_path + 'b.npy')

    result = (((result-b)/a) * std ) + mean
    msk = (((msk-b)/a) * std ) + mean
    img1 = np.zeros((256,256))
    img2 = np.zeros((256,256))

    msk1 = np.zeros((256,256))
    msk2 = np.zeros((256,256))
    #print(msk)
    for i in range(img1.shape[0]):
        for j in range(img2.shape[1]):
            if j == int(result[0,i]):
                img1[i,j] = 255

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if j == int(result[1,i]):
                img2[i,j] = 255
    
    cv2.imwrite('./data/img/result/extra/'+'resh0'+'.png',img1)
    cv2.imwrite('./data/img/result/extra/'+'resh1'+'.png',img2)

    print(asd)


    line1 = result[0,:]
    line2 = result[1,:]
    
    line_1.append(line1 + (left_coor[coun]-128))
    line_2.append(line2 + (right_coor[coun]-128))

    coun+=1
    if coun == 8 :
        coun = 0

line_1 = [item for sublist in line_1 for item in sublist]
line_2 = [item for sublist in line_2 for item in sublist]

img1 = np.zeros((2048,1403))
img2 = np.zeros((2048,1403))

img1_left = line_1[0:2048]
img1_right = line_2[0:2048]

img2_left = line_1[2048:4096]
img2_right = line_2[2048:4096]

for i in range(img1.shape[0]):
    for j in range(img2.shape[1]):
        if j == int(img1_left[i]) or j == int(img1_right[i]):
            img1[i,j] = 255

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if j == int(img2_left[i]) or j == int(img2_right[i]):
            img2[i,j] = 255

cv2.imwrite('./data/img/result/'+'label0'+'.png',img1)
cv2.imwrite('./data/img/result/'+'label1'+'.png',img2)
