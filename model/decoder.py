import tensorflow as tf
from tensorflow.keras import layers


class Decoder:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.decoder = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(self.latent_dim,)),
                layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                layers.Reshape(target_shape=(7, 7, 32)),
                layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same', activation='relu'
                ),
                layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=2, padding='same', activation='relu'
                ),
            ]
        )
