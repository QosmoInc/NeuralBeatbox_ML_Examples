import logging
import os
from typing import List, Any, Dict

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, \
    Concatenate, Flatten, Layer
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.models import Model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


class Sampling(Layer):
    def call(self, inputs: List[Tensor]) -> Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(mean=0.0, stddev=1.0, shape=(batch, dim))
        return z_mean + (tf.exp(0.5 * z_log_var) * epsilon)


def build_cond_mlp_enc(len_seq: int,
                       n_notes: int,
                       n_z: int,
                       dropout_rate: float = 0.40) -> Model:
    hidden_units = int(((2 * len_seq * n_notes) + n_z) / 2)

    onset_input = Input(shape=(len_seq, n_notes), name='onset_input')
    flat_onset_input = Flatten()(onset_input)
    vel_input = Input(shape=(len_seq, n_notes), name='vel_input')
    flat_vel_input = Flatten()(vel_input)

    cond_input = Input(shape=(n_notes,), name='cond_input')

    x = Concatenate()([flat_onset_input, flat_vel_input, cond_input])
    x = Dense(hidden_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    z_mean = Dense(n_z, activation='linear', name='z_mean')(x)
    z_log_var = Dense(n_z, activation='linear', name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])

    enc = Model([onset_input, vel_input, cond_input],
                [z_mean, z_log_var, z],
                name="cond_encoder")
    enc.summary()
    return enc


def build_cond_mlp_dec(len_seq: int,
                       n_notes: int,
                       n_z: int,
                       dropout_rate: float = 0.40) -> Model:
    hidden_units = int(((2 * len_seq * n_notes) + n_z) / 2)

    latent_input = Input(shape=(n_z,), name='z_sampling')
    cond_input = Input(shape=(n_notes,), name='cond_input')

    x = Concatenate()([latent_input, cond_input])
    x = Dense(hidden_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)

    onset_x = Dense((len_seq * n_notes), activation='sigmoid')(x)
    onset_output = Reshape((len_seq, n_notes), name='onset_output')(onset_x)
    vel_x = Dense((len_seq * n_notes), activation='relu')(x)
    vel_output = Reshape((len_seq, n_notes), name='vel_output')(vel_x)

    dec = Model([latent_input, cond_input],
                [onset_output, vel_output],
                name='cond_decoder')
    dec.summary()
    return dec


class RhythmVAE(Model):
    def __init__(self,
                 enc: Model,
                 dec: Model,
                 **kwargs: Any) -> None:
        super(RhythmVAE, self).__init__(**kwargs)
        self.enc = enc
        self.dec = dec

    def call(self, inputs: List[Tensor]) -> List[Tensor]:
        z_mean, z_log_var, z = self.enc(inputs)
        cond = inputs[-1]
        return self.dec([z, cond])

    def bce_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        loss = binary_crossentropy(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        return loss

    def mse_loss(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        loss = mean_squared_error(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        return loss

    def train_step(self,
                   data: (List[Tensor], List[Tensor])) -> Dict[str, float]:
        if isinstance(data, tuple):
            data = data[0]  # Only care about X since it's an autoencoder

        with tf.GradientTape() as tape:
            onset_data, vel_data, cond_data = data
            z_mean, z_log_var, z = self.enc(data)
            onset_rec, vel_rec = self.dec([z, cond_data])

            onset_rec_loss = self.bce_loss(onset_data, onset_rec)
            vel_rec_loss = self.mse_loss(vel_data, vel_rec)
            rec_loss = onset_rec_loss + vel_rec_loss

            kl_loss = 1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            total_loss = rec_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'onset_rec_loss': onset_rec_loss,
            'vel_rec_loss': vel_rec_loss,
        }

    def test_step(self,
                  data: (List[Tensor], List[Tensor])) -> Dict[str, float]:
        if isinstance(data, tuple):
            data = data[0]  # Only care about X since it's an autoencoder

        onset_data, vel_data, cond_data = data
        z_mean, z_log_var, z = self.enc(data)
        onset_rec, vel_rec = self.dec([z, cond_data])

        onset_rec_loss = self.bce_loss(onset_data, onset_rec)
        vel_rec_loss = self.mse_loss(vel_data, vel_rec)
        rec_loss = onset_rec_loss + vel_rec_loss

        kl_loss = 1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        total_loss = rec_loss + kl_loss

        return {
            'loss': total_loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss,
            'onset_rec_loss': onset_rec_loss,
            'vel_rec_loss': vel_rec_loss,
        }
