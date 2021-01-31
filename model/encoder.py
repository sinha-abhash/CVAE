import tensorflow as tf
from tensorflow.keras import layers


class Encoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(28,28,1)),
                layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'
                ),
                layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'
                ),
                layers.Flatten(),
                layers.Dense(self.latent_dim + self.latent_dim)
            ]
        )
