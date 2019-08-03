import tensorflow as tf 
import numpy as np 
import cv2

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

def bal_loss():
  def cost(labels, logits):
    loss = -tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits( labels=labels, logits=logits, pos_weight = 1, ))
    #loss = tf.keras.losses.binary_crossentropy(labels, logits)

    return loss
  return cost

dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.shuffle(3000)
dataset = dataset.batch(8)
dataset = dataset.repeat()
iterator = dataset.make_one_shot_iterator()
images, labels = iterator.get_next()

val_dataset = tf.data.TFRecordDataset('./data/record/val.tfrecords')
val_dataset = val_dataset.map(_parse_function)
#val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(20)
iterator_val = val_dataset.make_one_shot_iterator()
images_val, labels_val = iterator_val.get_next()

#print(tf.shape(images))

inputs = tf.keras.layers.Input(shape=(256,256,3))


x = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
x = tf.keras.layers.Conv2D(32,(3,3),padding='same', activation='relu', kernel_initializer='he_normal')(x)
x = tf.keras.layers.Conv2D(16,(3,3),padding='same', activation='relu', kernel_initializer='he_normal')(x)

predictions = tf.keras.layers.Conv2D(1,(3,3),padding='same', activation='sigmoid', kernel_initializer='he_normal')(x)

model = tf.keras.Model(inputs=inputs, outputs=predictions)


model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.001),
              loss=bal_loss(),
              metrics=['accuracy'])
              #target_tensors=[labels])

model.fit( images, labels,epochs=10, steps_per_epoch=40)

#model.evaluate(images_val, labels_val, steps=3)

result = model.predict(images_val, steps = 3)
#result = np.argmax(result, axis=3)
print(result)
result = result = np.where(result>0.5,1,0)
result = np.multiply( 255.0 , result)
print(result)

for i in range(len(result)):
  cv2.imwrite('./data/result/'+str(i)+'.png',result[i,:,:])  
