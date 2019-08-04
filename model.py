import gc
gc.collect()

import cv2
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

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

    return image_y, mask


dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(3000)
dataset = dataset.batch(8)
dataset = dataset.repeat()
#iterator = dataset.make_initializable_iterator()
#image, label = iterator.get_next()


val_dataset = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset = val_dataset.map(_parse_function)
val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(8)
#val_dataset = val_dataset.repeat()
#val_iterator = dataset.make_initializable_iterator()
#image_val, label_val = val_iterator.get_next()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(8,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(128,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(256,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(128,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(8,(3,3),padding='same', activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Conv2D(2,(3,3),padding='same', activation='sigmoid', kernel_initializer='he_normal')
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy])

#model.summary()

model.fit(dataset, epochs=25, steps_per_epoch=40)


model.summary()
#model.evaluate(val_dataset, steps=10)

result = model.predict(val_dataset) 
#res = result[0:1,:,:,:]
#print(result)

#result[:,:,:,0] = np.multiply((result[:,:,:,0]>0.5), result)
#result[:,:,:,1] = np.multiply((result[:,:,:,1]>0.5), result)
#res = np.reshape(res, (256,256,1))
#result = np.argmax(result, axis=3)

print(result.shape)
#print(result[0,:,:])
#result = np.multiply( 255.0 , result)
#print(result)
result = np.argmax(result, axis=3)
#print(result, result.shape)
#result = np.where(result==1,0,1)
result = np.multiply( 255.0 , result)
#print( result)
for i in range(len(result)):
  #img = np.uint8(result[i,:,:,:])
  """ cv2.namedWindow('image', cv2.WINDOW_NORMAL)
  cv2.imshow('image', result[i,:,:])
  cv2.waitKey(0)
  cv2.destroyAllWindows """
  cv2.imwrite('./data/result/'+str(i)+'.png',result[i,:,:])  

del model
gc.collect()

'''cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows'''