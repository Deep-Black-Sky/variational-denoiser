from model import DenoiserModel
from data import Data
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse

def kl_loss(dummy, concated_param):
    z_mean, z_log_var = tf.split(concated_param, num_or_size_splits=2, axis=1)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

def denoise_loss(y_true, y_pred):
    denoise_loss = mse(K.flatten(y_true), K.flatten(y_pred))
    return denoise_loss

def loss(y_true, y_pred, z_mean, z_log_var):
    kl = kl_loss(z_mean, z_log_var)
    denoise = denoise_loss(y_true, y_pred)
    loss = K.mean(denoise + kl)
    return loss

def train(iteration, epoch, num_sounds):
    model = DenoiserModel()
    vd = model.build_model(512)
    data = Data('cv-valid-dev.csv')

    vd.compile(optimizer='adam',
               loss=[kl_loss, denoise_loss],
               loss_weights=[1000.0, 1.0])

    for i in range(iteration):
        ys = data.make_batch(num_sounds)
        ys = np.reshape(ys, (ys.shape[0], 512, 1))
        xs = data.add_noise(ys, 0.0, 0.1)
        dummy = np.zeros(ys.shape[0], dtype=float)

        vd.fit(x=xs, y=[dummy, ys],
               epochs=epoch,
               validation_split=0.3)
