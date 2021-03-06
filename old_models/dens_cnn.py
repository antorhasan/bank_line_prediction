'''abandoned batch norm unet appraoch to predict next year bankline mask'''
import tensorflow as tf
import numpy as np
import cv2
from utils.crop import *
from datetime import datetime
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, InputLayer, Dense, Flatten, Reshape, Conv2DTranspose, concatenate
from tensorflow.keras.layers import BatchNormalization

tf.enable_eager_execution()


def _parse_function(example_proto):

    features = {
        "image_y": tf.FixedLenFeature((), tf.string),
        "image_m": tf.FixedLenFeature((), tf.string)
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)
    image_m = tf.decode_raw(parsed_features["image_m"],  tf.float64)

    image_y = tf.reshape(image_y, [256, 256, 6])
    image_m = tf.reshape(image_m, [256, 256, 1])
    image_y = tf.cast(image_y, dtype=tf.float32)
    image_m = tf.cast(image_m, dtype=tf.float32)

    image_y = tf.cast(image_y, tf.uint8)
    image_y = tf.cast(image_y, tf.float32)

    mean = np.load('./data/numpy_arrays/mean.npy')
    std = np.load('./data/numpy_arrays/std.npy')
    mean = tf.cast(mean, dtype=tf.float32)
    std = tf.cast(std, dtype=tf.float32)
    image_y = tf.math.divide(tf.math.subtract(image_y, mean), std)

    #image_y = tf.math.divide(tf.math.multiply(tf.cast(image_y,dtype=tf.int64), [255]), [3000])
    #image_y = tf.cast(image_y,dtype=tf.float32)

    img_lab = tf.cast(image_m, dtype=tf.bool)
    img_lab = tf.math.logical_not(img_lab)
    img_lab = tf.cast(img_lab, dtype=tf.float32)
    mask = tf.concat([image_m, img_lab], 2)

    return image_y, image_m


dataset = tf.data.TFRecordDataset('./data/record/train_tif.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(500)
dataset = dataset.batch(8)
#dataset = dataset.repeat()
#iterator = dataset.make_one_shot_iterator()
#images, labels = iterator.get_next()

val_dataset = tf.data.TFRecordDataset('./data/record/val_tif.tfrecords')
val_dataset = val_dataset.map(_parse_function)
#val_dataset = val_dataset.shuffle(3000)
val_dataset = val_dataset.batch(60)
#val_dataset = val_dataset.repeat()
#iterator_val = val_dataset.make_one_shot_iterator()
#images_val, labels_val = iterator_val.get_next()

logdir = "./data/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir, write_grads=True, write_images=True)


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch1 = BatchNormalization()
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch2 = BatchNormalization()
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch3 = BatchNormalization()
        self.pool3 = MaxPooling2D(pool_size=(2, 2))

        self.conv4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch4 = BatchNormalization()
        self.pool4 = MaxPooling2D(pool_size=(2, 2))

        self.conv5 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch5 = BatchNormalization()
        self.pool5 = MaxPooling2D(pool_size=(2, 2))

        self.conv6 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch6 = BatchNormalization()
        self.pool6 = MaxPooling2D(pool_size=(2, 2))

        self.conv7 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch7 = BatchNormalization()
        self.pool7 = MaxPooling2D(pool_size=(2, 2))

        """ self.up_convm = Conv2D(128, (2, 2), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch8 = BatchNormalization() """
        
        #self.flat = Flatten()
        self.reshape1 = Reshape((512,))
        self.dense = Dense(512,activation='relu',kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        #self.batch8 = BatchNormalization()
        self.reshape = Reshape((2, 2, 128)) 
        
        self.up_sam0 = UpSampling2D(size=(2, 2))
        self.up_conv0 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch9 = BatchNormalization()

        self.up_sam1 = UpSampling2D(size=(2, 2))
        self.up_conv1 = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch10 = BatchNormalization()

        self.up_sam2 = UpSampling2D(size=(2, 2))
        self.up_conv2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch11 = BatchNormalization()

        self.up_sam3 = UpSampling2D(size=(2, 2))
        self.up_conv3 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch12 = BatchNormalization()

        self.up_sam4 = UpSampling2D(size=(2, 2))
        self.up_conv4 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch13 = BatchNormalization()

        self.up_sam5 = UpSampling2D(size=(2, 2))
        self.up_conv5 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        self.batch14 = BatchNormalization()

        self.up_sam6 = UpSampling2D(size=(2, 2))
        self.up_conv6 = Conv2D(1, (3, 3), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01))
        """ self.batch15 = BatchNormalization()

        self.up_conv7 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                            bias_initializer=tf.keras.initializers.constant(.01)) """

    def call(self, inputs, training):
        i = self.conv1(inputs)
        
        i = self.batch1(i,training=training)
        x = self.pool1(i)
        a = self.conv2(x)

        a = self.batch2(a,training=training)
        x = self.pool2(a)
        s = self.conv3(x)
        
        s = self.batch3(s,training=training)
        x = self.pool3(s)
        d = self.conv4(x)
        
        d = self.batch4(d,training=training)
        x = self.pool4(d)
        f = self.conv5(x)
        
        f = self.batch5(f,training=training)
        x = self.pool5(f)
        g = self.conv6(x)
        
        g = self.batch6(g,training=training)
        x = self.pool6(g)
        e = self.conv7(x)
        
        e = self.batch7(e,training=training)
        x = self.pool7(e)

        x = self.reshape1(x)
        x = self.dense(x)
        x = self.reshape(x)
        #x = self.up_convm(x)
        
        x = self.up_sam0(x)
        #x = concatenate([e,x])
        x = self.up_conv0(x)
        
        x = self.batch9(x,training=training)
        x = self.up_sam1(x)
        #x = concatenate([g,x])
        x = self.up_conv1(x)
        
        x = self.batch10(x,training=training)
        x = self.up_sam2(x)
        #x = concatenate([f,x])
        x = self.up_conv2(x)
        
        x = self.batch11(x,training=training)
        x = self.up_sam3(x)
        #x = concatenate([d,x])
        x = self.up_conv3(x)
        
        x = self.batch12(x,training=training)
        x = self.up_sam4(x)
        #x = concatenate([s,x])
        x = self.up_conv4(x)
        
        x = self.batch13(x,training=training)
        x = self.up_sam5(x)
        #x = concatenate([a,x])
        x = self.up_conv5(x)
        
        x = self.batch14(x,training=training)
        x = self.up_sam6(x)
        #x = concatenate([i,x])
        outputs = self.up_conv6(x)
        """ if training:
            x = self.batch15(x,training=training)
        outputs = self.up_conv7(x) """
        return outputs

    def model(self):
        x = tf.keras.layers.Input(shape=(256, 256, 6))

        return tf.keras.Model(inputs=[x], outputs=self.call(x,training=True)).summary()


model = MyModel()

model.model()


def loss_object(labels, predictions):
    #loss = tf.reduce_mean(-0.93*tf.math.multiply(labels,tf.math.log(predictions))-(1-0.93)*tf.math.multiply((1-labels),tf.math.log(1-predictions)))

    loss = -0.7*tf.reduce_mean(tf.math.multiply(labels, tf.math.log(predictions)))-(1-0.7)*tf.reduce_mean(tf.math.multiply((1-labels), tf.math.log(1-predictions)))
    #loss1 = tf.losses.absolute_difference(labels, predictions)

    #loss = tf.losses.hinge_loss(labels, predictions)
    return loss


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')

optimizer = tf.keras.optimizers.Adam(learning_rate=.00001)


@tf.function
def train_step(images, labels):

    with tf.GradientTape() as tape:
        predictions = model(images,training=True)
        loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images,training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def predict_step(images):
    result = model(images,training=False)
    result = result = np.where(result > 0.5, 1, 0)
    result = np.multiply(255.0, result)

    for i in range(len(result)):
        cv2.imwrite('./data/result/'+str(i)+'.png', result[i, :, :])

    stitch_imgs()


EPOCHS = 25

for epoch in range(EPOCHS):
    for images, labels in dataset:
        train_step(images, labels)

    for test_images, test_labels in val_dataset:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


for test_images, test_labels in val_dataset:
    predict_step(test_images)

for images, labels in dataset:
    result = model(images,training=False)
    result = result = np.where(result > 0.5, 1, 0)
    result = np.multiply(255.0, result)

    for i in range(len(result)):
        cv2.imwrite('./data/result/stitched/label/pred/' +
                    str(i)+'.png', result[i, :, :])

    result = np.multiply(255.0, labels)
    for i in range(len(labels)):
        cv2.imwrite('./data/result/stitched/label/' +
                    str(i)+'.png', result[i, :, :])
    break


'''
model = cnn_model()

model.model()

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=.001),
              loss=f_loss(),
              metrics=['accuracy'],)

model.fit( images, labels,epochs=2, steps_per_epoch=40, validation_data= val_dataset,
          validation_steps=3, callbacks=[tensorboard_callback])


'''

""" result = model.predict(val_dataset, steps = 1)
result = result = np.where(result>0.5,1,0)
result = np.multiply( 255.0 , result)

for i in range(len(result)):
	cv2.imwrite('./data/result/'+str(i)+'.png',result[i,:,:])

stitch_imgs()  """
