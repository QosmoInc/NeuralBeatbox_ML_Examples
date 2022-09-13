import logging
import os
from typing import List, Union

import numpy as np
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from rhythm.models_mlp import build_cond_mlp_enc, build_cond_mlp_dec, RhythmVAE
from rhythm.models_rnn import build_lstm_enc, build_lstm_dec
from rhythm.preprocessing import N_DRUMS, calc_conditioning_data, LEN_SEQ

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def load_rhythm_vae_data(paths: List[str]) -> (np.ndarray, np.ndarray):
    onset_xs = []
    vel_xs = []

    for path in paths:
        onset_x = np.load(path)['drum_onset_matrices']
        vel_x = np.load(path)['drum_vel_matrices']
        onset_xs.append(onset_x)
        vel_xs.append(vel_x)
        log.info(f'{path} onset_x shape = {onset_x.shape}')
        log.info(f'{path} vel_x shape = {vel_x.shape}')

    onset_x = np.vstack(onset_xs)
    vel_x = np.vstack(vel_xs)
    log.info(f'Final onset_x shape = {onset_x.shape}')
    log.info(f'Final vel_x shape = {vel_x.shape}')

    return onset_x, vel_x


def train_rhythm_vae(
        vae: Model,
        x: Union[np.ndarray, List[np.ndarray]],
        y: Union[np.ndarray, List[np.ndarray]],
        model_name: str,
        batch_size: int = 64,
        epochs: int = 50,
        val_split: float = 0.10,
        patience: int = 10,
        output_dir_path: str = '../../models') -> None:
    save_path = os.path.join(
        output_dir_path, model_name + '_e{epoch:02d}_vl{val_loss:.4f}.h5')
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=patience,
                       verbose=1)
    cp = ModelCheckpoint(save_path,
                         monitor='val_loss',
                         verbose=1,
                         save_weights_only=True,
                         save_best_only=True)
    vae.fit(x,
            y,
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=val_split,
            callbacks=[es, cp],
            verbose=1)


def rhythm_vae_training_example(training_data_paths: List[str]) -> None:
    n_z = 128
    len_seq = LEN_SEQ
    n_notes = N_DRUMS
    batch_size = 64
    epochs = 50
    val_split = 0.10
    dropout = 0.4
    # model_name = f'vae_mlp_cond_nz{n_z}_do{int(dropout * 100)}'
    model_name = f'vae_lstm_nz{n_z}'
    binary_cond_data = False
    use_zscore = True

    onset_x, vel_x = load_rhythm_vae_data(training_data_paths)
    cond_data, cond_mean, cond_std = calc_conditioning_data(
        onset_x, binary_only=binary_cond_data, use_zscore=use_zscore)
    log.info(f"cond_data.shape = {cond_data.shape}")

    # data = np.load("/Users/puntland/local_christhetree/qosmo/NeuralBeatbox_ML_Examples/data/rhythm/combine-001.npz")
    # cvec = np.load("/Users/puntland/local_christhetree/qosmo/NeuralBeatbox_ML_Examples/data/rhythm/cvec.npz")
    # cvec = cvec["arr_0"]
    # ts_onset = data["drum_onset_matrices"]
    # ts_vel = data["drum_vel_matrices"]
    # stats = np.sum(cvec, axis=0)
    # exit()

    # enc = build_cond_mlp_enc(len_seq, n_notes, n_z, dropout)
    enc = build_lstm_enc(n_z=n_z, n_drums=n_notes, len_seq=len_seq, n_cond=8)
    # dec = build_cond_mlp_dec(len_seq, n_notes, n_z, dropout)
    dec = build_lstm_dec(n_z=n_z, n_drums=n_notes, len_seq=len_seq, n_cond=8)

    vae = RhythmVAE(enc, dec)
    vae.compile(optimizer='adam')

    train_rhythm_vae(vae,
                     [onset_x, vel_x, cond_data],
                     [onset_x, vel_x],
                     model_name,
                     batch_size,
                     epochs,
                     val_split)
