import logging
import os

from keras import Model
from keras.layers import Input, Dense, LSTM, Concatenate, RepeatVector, TimeDistributed, Bidirectional

from rhythm.models_mlp import Sampling

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def build_lstm_enc(n_z: int = 512,
                   lstm_size: int = 512,
                   n_drums: int = 9,
                   len_seq: int = 168,
                   is_conditional: bool = True,
                   n_cond: int = 9) -> Model:
    d_input = Input(shape=(len_seq, n_drums), name='d_input')
    d_vel_input = Input(shape=(len_seq, n_drums), name='d_vel_input')
    cond_input = Input(shape=(n_cond,), name='cond_input')

    inputs = [d_input, d_vel_input]
    lstm_in = Concatenate(axis=-1, name='lstm_in')([d_input, d_vel_input])

    level_1_lstm = Bidirectional(LSTM(lstm_size, return_sequences=False), name='level_1_enc_lstm')
    level_1_emb = level_1_lstm(lstm_in)

    cond_flag = ''
    if is_conditional:
        inputs.append(cond_input)
        cond_flag = 'cond_'
        level_1_emb = Concatenate()([level_1_emb, cond_input])

    z_mean = Dense(n_z, activation='linear', name='z_mean')(level_1_emb)
    z_log_var = Dense(n_z, activation='linear', name='z_log_var')(level_1_emb)
    z = Sampling()([z_mean, z_log_var])

    enc = Model(inputs,
                [z_mean, z_log_var, z],
                name=f'{cond_flag}lstm_enc')
    enc.summary()
    return enc


def build_lstm_dec(n_z: int = 512,
                   lstm_size: int = 512,
                   n_drums: int = 9,
                   len_seq: int = 168,
                   is_conditional: bool = True,
                   n_cond: int = 9) -> Model:
    latent_input = Input(shape=(n_z,), name='z_sampling')
    cond_input = Input(shape=(n_cond,), name='cond_input')

    if is_conditional:
        z = Concatenate()([latent_input, cond_input])
    else:
        z = latent_input

    repeated_z = RepeatVector(len_seq)(z)
    x = LSTM(lstm_size,
             return_sequences=True,
             name='level_1_dec_lstm')(repeated_z)
    d_sigmoid = TimeDistributed(Dense(n_drums, activation='sigmoid'), name='d_sigmoid')
    d_vel_relu = TimeDistributed(Dense(n_drums, activation='relu'), name='d_vel_relu')
    d_out = d_sigmoid(x)
    d_vel_out = d_vel_relu(x)

    inputs = [latent_input]

    cond_flag = ''
    if is_conditional:
        inputs.append(cond_input)
        cond_flag = 'cond_'

    dec = Model(inputs,
                [d_out, d_vel_out],
                name=f'{cond_flag}lstm_dec')
    dec.summary()
    return dec
