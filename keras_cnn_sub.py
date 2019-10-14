import tensorflow as tf 
import numpy as np 
import cv2
from utils.crop import *
from datetime import datetime


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
        self.conv1 = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        
        self.conv2 = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.conv3 = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
        
        self.conv4 = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))

        self.conv5 = tf.keras.layers.Conv2D(1,(1,1),padding='same', activation='sigmoid', kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.constant(.01))
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        outputs = self.conv5(x)

        return outputs

model = cnn_model()


model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.001),
              loss=f_loss(),
              metrics=['accuracy'],)

model.fit( images, labels,epochs=15, steps_per_epoch=40, validation_data= val_dataset,
          validation_steps=3, callbacks=[tensorboard_callback])

model.summary()

result = model.predict(images_val, steps = 1)
result = result = np.where(result>0.5,1,0)
result = np.multiply( 255.0 , result)

for i in range(len(result)):
  cv2.imwrite('./data/result/'+str(i)+'.png',result[i,:,:])

stitch_imgs()