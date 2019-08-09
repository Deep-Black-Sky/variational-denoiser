from model import DenoiserModel
from data import Data
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
import IPython

def kl_loss(dummy, concated_param):
    z_mean, z_log_var = tf.split(concated_param, num_or_size_splits=2, axis=1)
    kl_loss = 1.0 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(K.abs(kl_loss) * 1000.0)

def denoise_loss(ys_xs_noise, y_pred):
    y_true, mixed, noise = tf.split(ys_xs_noise, num_or_size_splits=3, axis=1)
    noise_pred = tf.subtract(mixed, y_pred)
    denoise_loss = tf.add(mse(y_true, y_pred), mse(noise, noise_pred))
    return K.mean(denoise_loss)

def loss(y_true, y_pred, z_mean, z_log_var):
    kl = kl_loss(z_mean, z_log_var)
    denoise = denoise_loss(y_true, y_pred)
    loss = K.mean(denoise + kl)
    return loss

def train(iteration, epoch, num_sounds, is_jupyter):
    model = DenoiserModel()
    vd = model.build_model(512)
    data = Data('cv-valid-dev.csv')

    vd.compile(optimizer='adam',
               loss=[kl_loss, denoise_loss],
               loss_weights=[1.0, 1.0])

    for i in range(iteration):
        ys = data.make_batch(num_sounds)
        ys = np.reshape(ys, (ys.shape[0], 512, 1))
        xs, noise = data.add_noise(ys, 0.0, 0.08)
        ys_xs_noise = np.concatenate([ys, xs, noise], axis=1)
        dummy = np.zeros(ys_xs_noise.shape[0], dtype=float)

        vd.fit(x=xs, y=[dummy, ys_xs_noise],
               epochs=epoch,
               validation_split=0.3)

        if is_jupyter:
            test = data.make_batch(1)
            IPython.display.display(IPython.display.Audio(data=test.flatten(), rate=48000))
            test = np.reshape(test, (test.shape[0], 512, 1))
            test, noise = data.add_noise(test, 0.0, 0.08)
            IPython.display.display(IPython.display.Audio(data=test.flatten(), rate=48000))
            denoised = denoise(vd, test)
            IPython.display.display(IPython.display.Audio(data=denoised, rate=48000))

def denoise(model, data):
    denoised = model.predict(data)
    denoised = denoised[1]
    denoised = np.array(denoised).flatten()
    return denoised

if __name__ == '__main__':
    train(50, 10, 1, False)
