from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D, Flatten, Dense, \
    Input, Lambda, Activation, Reshape, Multiply, Add, Concatenate
from tensorflow.keras import backend as K

def wavenetBlock(n_atrous_filters, atrous_filter_size, atrous_rate):
    def f(input_):
        residual = input_
        tanh_out = Convolution1D(n_atrous_filters, atrous_filter_size,
                                 dilation_rate=atrous_rate,
                                 padding='same',
                                 activation='tanh')(input_)
        sigmoid_out = Convolution1D(n_atrous_filters, atrous_filter_size,
                                    dilation_rate=atrous_rate,
                                    padding='same',
                                    activation='sigmoid')(input_)
        merged = Multiply()([tanh_out, sigmoid_out])
        skip_out = Convolution1D(1, 1, activation='relu', padding='same')(merged)
        out = Add()([skip_out, residual])
        return out, skip_out
    return f

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class DenoiserModel:
    encoder = None
    denoiser = None
    VD = None

    def build_model(self, input_size):
        input = Input(shape=(input_size, 1))
        # WaveNet Variational Encoder
        out, skip_out = wavenetBlock(64, 2, 2)(input)
        skip_connections = [skip_out]
        for i in range(20):
            out, skip_out = wavenetBlock(64, 2, 2**((i+2)%9))(out)
            skip_connections.append(skip_out)
        encoder_net = Add()(skip_connections)
        encoder_net = Activation('relu')(encoder_net)
        encoder_net = Convolution1D(1, 1, activation='relu')(encoder_net)
        encoder_net = Convolution1D(1, 1)(encoder_net)
        encoder_net = Flatten()(encoder_net)
        z_mean = Dense(input_size)(encoder_net)
        z_log_var = Dense(input_size)(encoder_net)
        z = Lambda(sampling, output_shape=(input_size, 1))([z_mean, z_log_var])

        # WaveNet Denoiser
        z = Reshape(target_shape=(input_size, 1))(z)
        denoiser_input = Concatenate()([input, z])
        out, skip_out = wavenetBlock(64, 2, 2)(denoiser_input)
        skip_connections = [skip_out]
        for i in range(20):
            out, skip_out = wavenetBlock(64, 2, 2**((i+2)%9))(out)
            skip_connections.append(skip_out)
        denoiser_net = Add()(skip_connections)
        denoiser_net = Activation('relu')(denoiser_net)
        denoiser_net = Convolution1D(1, 1, activation='relu')(denoiser_net)
        denoiser_net = Convolution1D(1, 1)(denoiser_net)
        denoiser_net = Flatten()(denoiser_net)
        denoiser_net = Dense(input_size, activation='softmax')(denoiser_net)
        denoiser_net = Reshape(target_shape=(input_size, 1), name='denoise')(denoiser_net)

        model = Model(inputs=input,
                      outputs=[Concatenate(name='kl')([z_mean, z_log_var]), denoiser_net])
                      # Concat z_mean with z_log_var because
                      # fit function needs the loss function per losses.
                      # We need to culc two losses which are kl_loss and denoise_loss.

        # model.summary()

        return model
