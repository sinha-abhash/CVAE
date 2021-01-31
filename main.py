import tensorflow as tf
import time
import logging

from model import VAE
import config
from utils import utils
from data_preprocess import Data

logging.basicConfig(level=logging.INFO)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_and_test():
    # pick a test sample
    dataset = Data(config.train_size, config.batch_size, config.test_size)
    for test_batch in dataset.test_dataset:
        test_sample = test_batch[:config.num_examples_to_generate, :, :, :]

    random_vector_for_generation = tf.random.normal(shape=[config.num_examples_to_generate, config.latent_dim])
    model = VAE(config.latent_dim)

    #utils.generate_and_save_images(model, 0, test_sample)

    optimizer = tf.keras.optimizers.Adam(1e-4)
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        for train_x in dataset.train_dataset:
            utils.train_step(model, train_x, optimizer=optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in dataset.test_dataset:
            loss(utils.compute_loss(model, test_x))

        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        utils.generate_and_save_images(model, epoch, test_sample)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_and_test()

