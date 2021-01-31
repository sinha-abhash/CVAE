import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


class Data:
    def __init__(self, train_size, batch_size, test_size):
        (self.train_images, _), (self.test_images, _) = tf.keras.datasets.mnist.load_data()
        self.logger = logging.getLogger("preprocess images")

        self.train_size = train_size
        self.batch_size = batch_size
        self.test_size = test_size

        self.train_images = self.preprocess(self.train_images, "train")
        self.test_images = self.preprocess(self.test_images, "test")

        self.train_dataset = None
        self.test_dataset = None

        self.create_batch()

    def preprocess(self, images, phase):
        self.logger.info(f"Preprocessing: {phase} images")
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')

    def create_batch(self):
        self.logger.info("Creating batch")
        self.train_dataset = (tf.data.Dataset.from_tensor_slices(self.train_images).shuffle(self.train_size).batch(self.batch_size))
        self.test_dataset = (tf.data.Dataset.from_tensor_slices(self.test_images).shuffle(self.test_size).batch(self.batch_size))


if __name__ == '__main__':
    data = Data(10, 2, 3)
