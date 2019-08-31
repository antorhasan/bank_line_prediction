import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Conv2D, CuDNNLSTM, LSTM, Dense


tf.enable_eager_execution()


def _parse_function(example_proto):

    features = {
            "image_y": tf.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.int64)

    image_y = tf.cast(image_y, dtype=tf.float32)

    mean = np.load('./data/numpy_arrays/thin_line/mean.npy')
    std = np.load('./data/numpy_arrays/thin_line/std.npy')
    a = np.load('./data/numpy_arrays/thin_line/a.npy')
    b = np.load('./data/numpy_arrays/thin_line/b.npy')

    image_y = (image_y-mean)/std

    image_y = (image_y*a) + b  

    return image_y


dataset = tf.data.TFRecordDataset('./data/record/thin/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.window(size=28, shift=28, stride=1,drop_remainder=False).flat_map(lambda x: x.batch(28))
dataset = dataset.map(lambda x: tf.data.Dataset.from_tensor_slices(x))
dataset = dataset.flat_map(lambda x: x.window(size=3, shift=1, stride=1,drop_remainder=True))
dataset = dataset.flat_map(lambda x: x.batch(3))

val_dataset = tf.data.TFRecordDataset('./data/record/val_tif.tfrecords')
val_dataset = val_dataset.map(_parse_function)

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.lstm = LSTM(10)
        self.dense = Dense(1)

    def call(self, inputs):
        x = self.lstm(inputs)
        x = self.dense(x)
        return x
    
    def model(self):
        x = tf.keras.layers.Input(shape=(3, 1))

        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


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

	test_loss(t_loss)
	test_accuracy(labels, predictions)


model = MyModel()

model.model()


optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)
#loss_object = tf.keras.losses.mse(labels, predictions)

def loss_object(labels, predictions):
	loss = tf.keras.losses.mse(labels, predictions)
	return loss


train_loss = tf.keras.metrics.MeanSquaredError()
#train_accuracy = tf.keras.metrics.Accuracy()

EPOCHS = 50

for epoch in range(EPOCHS):
	for data in dataset:
		train_step(data[:,0:2,:], data[:,2:3,:])


	template = 'Epoch {}, Loss: {}'
	print(template.format(epoch+1,
						train_loss.result()))

	# Reset the metrics for the next epoch
	train_loss.reset_states()
	#train_accuracy.reset_states()
