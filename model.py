import tensorflow as tf
from tensorflow.keras import layers
import cv2
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

    return image_y,image_m


dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(3000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
image, label = iterator.get_next()


val_dataset = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset = val_dataset.map(_parse_function)
val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(8)
val_dataset = val_dataset.repeat()
val_iterator = dataset.make_initializable_iterator()
image_val, label_val = val_iterator.get_next()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),
  tf.keras.layers.Conv2D(1,(3,3),padding='same', activation='sigmoid')
])

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.metrics.mean_per_class_accuracy])

model.summary()

model.fit(image, label, epochs=2, steps_per_epoch=20)



'''model.evaluate(val_dataset, steps=10)

result = model.predict(val_dataset, steps=2) """
#res = result[0:1,:,:,:]
#res = np.multiply((result>0.5), result)
#res = np.reshape(res, (256,256,1))

#print(res.shape)
#print(res)
""" for i in range(len(res)):
  img = np.uint8(res[i,:,:,:])
  cv2.imwrite('./data/result/'+str(i)+'.png',img) """

""" cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', res)
cv2.waitKey(0)
cv2.destroyAllWindows'''