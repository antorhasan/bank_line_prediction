import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Conv2D, CuDNNLSTM, LSTM, Dense
import cv2
from datetime import datetime

tf.enable_eager_execution()


def _parse_function(example_proto):

    features = {
            "image_y": tf.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float32)

    image_y = tf.cast(image_y, dtype=tf.float32)

    """ mean = np.load('./data/numpy_arrays/thin_line/mean.npy')
    std = np.load('./data/numpy_arrays/thin_line/std.npy')
    a = np.load('./data/numpy_arrays/thin_line/a.npy')
    b = np.load('./data/numpy_arrays/thin_line/b.npy')

    image_y = (image_y-mean)/std

    image_y = (image_y*a) + b  """ 

    return image_y


class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        
        self.lstm = LSTM(100,bias_initializer=tf.keras.initializers.constant(.01),activation='relu')
        self.dense1 = Dense(100, activation='relu',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')
        self.dense2 = Dense(1, activation='tanh',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.lstm(inputs)
        #x = self.dense1(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def model(self):
        x = tf.keras.layers.Input(shape=(27, 1))

        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()

def loss_object(labels, predictions):
    loss = tf.keras.losses.mse(labels, predictions)
    return loss


@tf.function
def train_step(images, labels):

    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(labels, predictions)
    #train_accuracy(labels, predictions)

@tf.function
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


dataset = tf.data.TFRecordDataset('./data/record/normal_dis/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.window(size=28, shift=28, stride=1,drop_remainder=False).flat_map(lambda x: x.batch(28))
""" dataset = dataset.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset = dataset.flat_map(lambda x: x.window(size=3, shift=1, stride=1,drop_remainder=True))
dataset = dataset.flat_map(lambda x: x.batch(3)) """
#dataset = dataset.shuffle(3000)
#dataset = dataset.batch(1)


val_dataset = tf.data.TFRecordDataset('./data/record/normal_dis/val28.tfrecords')
val_dataset = val_dataset.map(_parse_function)
val_dataset = val_dataset.window(size=29, shift=29, stride=1,drop_remainder=False).flat_map(lambda x: x.batch(29))
val_dataset = val_dataset.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
val_dataset = val_dataset.flat_map(lambda x: x.window(size=28, shift=1, stride=1,drop_remainder=True))
val_dataset = val_dataset.flat_map(lambda x: x.batch(28))
val_dataset = val_dataset.batch(32)

test = tf.data.TFRecordDataset('./data/record/normal_dis/val28.tfrecords')
test = test.map(_parse_function)
test = test.window(size=29, shift=29, stride=1,drop_remainder=False).flat_map(lambda x: x.batch(29))
test = test.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
test = test.flat_map(lambda x: x.window(size=28, shift=1, stride=1,drop_remainder=True))
test = test.flat_map(lambda x: x.batch(28))
test = test.batch(5404)

model = MyModel()

model.model()

optimizer = tf.keras.optimizers.Adam(learning_rate=.00001)
#min lr which is working = .0001 for 1 unit lstm

#loss_object = tf.keras.losses.mse(labels, predictions)

train_loss = tf.keras.metrics.MeanSquaredError()
#train_accuracy = tf.keras.metrics.Accuracy()
test_loss = tf.keras.metrics.MeanSquaredError()

#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

logdir = "./data/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, write_grads=True, write_images=True)
callback = [tensorboard_callback]
tensorboard_callback.set_model(model)
#model.evaluate(callbacks=callback)

EPOCHS = 2
for epoch in range(EPOCHS):
    for data in dataset:
        train_step(data[:, 0:27, :], data[:, 27:28, :])

    for data_val in val_dataset:
        test_step(data_val[:, 0:27, :], data_val[:, 27:28, :])
    
    template = 'Epoch {}, Loss: {}, Test Loss: {},'
    print(template.format(epoch+1,
                        train_loss.result(), test_loss.result() ))

    tensorboard_callback.on_epoch_end(epoch, logs={'train_loss':train_loss,'test_loss':test_loss})
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    test_loss.reset_states()

	#train_accuracy.reset_states()

for data in test:
    predict_step(data[:, 0:27, :])
