import tensorflow as tf
from tensorflow.keras import layers

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

val_dataset = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset = val_dataset.map(_parse_function)
val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(8)
val_dataset = val_dataset.repeat()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='relu'),
  tf.keras.layers.Conv2D(64,(3,3),padding='same', activation='sigmoid')
])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.categorical_accuracy])


model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
