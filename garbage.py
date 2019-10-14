class data_fetch():
    
    def __init__(self, batch):
        self.batch = batch
        self.dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
        print(self.dataset.output_shapes)
        self.dataset = self.dataset.map(self._parse_function)
        self.dataset = self.dataset.shuffle(1000)
        self.dataset = self.dataset.batch(self.batch)
        self.dataset = self.dataset.repeat()

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
