import tensorflow as tf 
import numpy as np 
import cv2
from utils.crop import *
from datetime import datetime
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, InputLayer, Dense, Flatten, Reshape, Conv2DTranspose


def _parse_function(example_proto):

    features = {
                "image_y": tf.FixedLenFeature((), tf.string ),
                "image_m": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)
    image_m = tf.decode_raw(parsed_features["image_m"],  tf.float64)

    image_y = tf.reshape(image_y, [256,256,3])
    image_m = tf.reshape(image_m, [256,256,1])
    image_y = tf.cast(image_y,dtype=tf.float32)
    image_m = tf.cast(image_m,dtype=tf.float32)

    img_lab = tf.cast(image_m,dtype=tf.bool)
    img_lab = tf.math.logical_not( img_lab )
    img_lab = tf.cast(img_lab,dtype=tf.float32)
    mask = tf.concat([image_m, img_lab], 2)

    return image_y, image_m

def f_loss():
  def cost(labels, logits):
    loss = tf.reduce_mean(-0.95*tf.math.multiply(labels,tf.math.log(logits))-(1-0.95)*tf.math.multiply((1-labels),tf.math.log(1-logits)))
    return loss
  return cost

dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(8)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

val_dataset = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset = val_dataset.map(_parse_function)
#val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(60)
val_dataset = val_dataset.repeat()
iterator_val = val_dataset.make_one_shot_iterator()
images_val, labels_val = iterator_val.get_next()

logdir = "./data/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_grads=True, write_images=True)


class cnn_model(tf.keras.Model):

    def __init__(self):
        super(cnn_model, self).__init__()
        #self.input1 = InputLayer(input_shape= (256,256,3))
        self.conv1 = Conv2D(8,(3,3),input_shape=(256,256,3), padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.conv2 = Conv2D(8,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.conv4 = Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv5 = Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.conv6 = Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.pool3 = MaxPooling2D(pool_size=(3, 3))


        self.conv7 = Conv2D(48,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.conv8 = Conv2D(48,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.pool4 = MaxPooling2D(pool_size=(3, 3))

        self.conv9 = Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.conv10 = Conv2D(48,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.pool5 = MaxPooling2D(pool_size=(3, 3))

        self.flat = Flatten()
        self.dense1 = Dense(192)
        self.dense2 = Dense(192)
        self.dense3 = Dense(192)
        self.reshape = Reshape((2,2,48))

        self.up_sam1 = UpSampling2D(size = (3,3))
        self.up_conv1 = Conv2DTranspose(48,(3,3),padding='valid', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.up_conv2 = Conv2DTranspose(64,(3,3),padding='valid', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.up_sam2 = UpSampling2D(size = (3,3))
        self.up_conv3 = Conv2DTranspose(48,(3,3),padding='valid', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.up_conv4 = Conv2D(48,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.up_sam3 = UpSampling2D(size = (2,2))
        self.up_conv5 = Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.up_conv6 = Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        
        self.up_sam4 = UpSampling2D(size = (2,2))
        self.up_conv7 = Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.up_conv8 = Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.up_sam5 = UpSampling2D(size = (2,2))
        self.up_conv9 = Conv2DTranspose(8,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        self.up_conv10 = Conv2DTranspose(8,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.up_conv11 = Conv2D(1,(1,1),padding='same', activation='sigmoid', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool4(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool5(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape(x)
        x = self.up_sam1(x)
        x = self.up_conv1(x)
        x = self.up_conv2(x)
        x = self.up_sam2(x)
        x = self.up_conv3(x)
        x = self.up_conv4(x)
        x = self.up_sam3(x)
        x = self.up_conv5(x)
        x = self.up_conv6(x)
        x = self.up_sam4(x)
        x = self.up_conv7(x)
        x = self.up_conv8(x)
        x = self.up_sam5(x)
        x = self.up_conv9(x)
        x = self.up_conv10(x)
        outputs = self.up_conv11(x)
    
        return outputs

    def model(self):
        x = tf.keras.layers.Input(shape=(256, 256, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


model = cnn_model()

model.model()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.00001),
              loss=f_loss(),
              metrics=['accuracy'])

model.fit( images, labels,epochs=1000, steps_per_epoch=40, validation_data= val_dataset,
          validation_steps=3, callbacks=[tensorboard_callback])



result = model.predict(images_val, steps = 1)
result = result = np.where(result>0.5,1,0)
result = np.multiply( 255.0 , result)

for i in range(len(result)):
  cv2.imwrite('./data/result/'+str(i)+'.png',result[i,:,:])

stitch_imgs()