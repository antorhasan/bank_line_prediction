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

def bin_loss():
  def cost(labels, logits):
    #loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( labels=labels, logits=logits, pos_weight = 3)) #no luck
    #loss = tf.keras.losses.binary_crossentropy(labels, logits)  #no luck with dilated ,should abandon this
    """ loss = tf.losses.absolute_difference(
      labels,
      logits,
      weights=1.0,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS) """
    #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)   #with relu no luck at 15 epochs
    return loss
  return cost

def f_loss():
  def cost(labels, logits):
    #kernel = np.ones((7,7), np.uint8)
    #labels = cv2.dilate(labels, kernel, iterations=1)
    #wt = 0.6
    loss = tf.reduce_mean(-0.9*tf.math.multiply(labels,tf.math.log(logits))-(1-0.9)*tf.math.multiply((1-labels),tf.math.log(1-logits)))
    #loss2 = tf.losses.absolute_difference(labels, logits)
    #loss3 = tf.losses.hinge_loss(labels, logits)
    return loss
  return cost

dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.shuffle(3000)
dataset = dataset.batch(8)
dataset = dataset.repeat()
#iterator = dataset.make_one_shot_iterator()
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
images, labels = iterator.get_next()

val_dataset = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset = val_dataset.map(_parse_function)
#val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(20)
#iterator_val = val_dataset.make_one_shot_iterator()
iterator_val = tf.compat.v1.data.make_one_shot_iterator(val_dataset)
images_val, labels_val = iterator_val.get_next()

""" val_dataset1 = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset1 = val_dataset1.map(_parse_function)
#val_dataset = val_dataset.shuffle(3000)
val_dataset1 = val_dataset1.batch(20)
iterator_val1 = val_dataset1.make_one_shot_iterator()
images_val1, labels_val1 = iterator_val1.get_next()
 """
#print(tf.shape(images))

inputs = tf.keras.layers.Input(shape=(256,256,3))

logdir = "./data/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_grads=True, write_images=True)

x = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
    bias_initializer=tf.keras.initializers.constant(.01))(inputs)
x = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
    bias_initializer=tf.keras.initializers.constant(.01))(x)
#x = tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
#    bias_initializer=tf.keras.initializers.constant(.01))(x)
#x = tf.keras.layers.Conv2D(128,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
#    bias_initializer=tf.keras.initializers.constant(.01))(x)
#x = tf.keras.layers.Conv2D(256,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
#    bias_initializer=tf.keras.initializers.constant(.01))(x)
#x = tf.keras.layers.Conv2D(128,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
#    bias_initializer=tf.keras.initializers.constant(.01))(x)
#x = tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
#    bias_initializer=tf.keras.initializers.constant(.01))(x)
x = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
    bias_initializer=tf.keras.initializers.constant(.01))(x)
x = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal',
    bias_initializer=tf.keras.initializers.constant(.01))(x)

predictions = tf.keras.layers.Conv2D(1,(1,1),padding='same', activation='sigmoid', kernel_initializer='he_normal',
    bias_initializer=tf.keras.initializers.constant(.01))(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.001),
              loss=f_loss(),
              metrics=['accuracy'],)
              #target_tensors=[labels])

model.fit( images, labels,epochs=15, steps_per_epoch=40, validation_data= val_dataset,
          validation_steps=3, callbacks=[tensorboard_callback])

#model.evaluate(images_val, labels_val, steps=3)

result = model.predict(images_val, steps = 3)
#result = np.argmax(result, axis=3)
#print(result)
result = result = np.where(result>0.5,1,0)
#result = np.multiply( 255.0 , result)
#result = np.argmax(result, axis=3)
#print(result,result.shape)
result = np.multiply( 255.0 , result)

for i in range(len(result)):
  cv2.imwrite('./data/result/'+str(i)+'.png',result[i,:,:])

stitch_imgs()