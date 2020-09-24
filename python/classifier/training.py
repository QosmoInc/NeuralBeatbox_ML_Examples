import logging
import os
from typing import Dict, Tuple

import numpy as np
from sklearn.utils import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from classifier.models import build_cnn3_classifier

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def train_drum_classifier(
        classifier: Model,
        x: np.ndarray,
        y: np.ndarray,
        model_name: str,
        batch_size: int = 32,  # Small batch size provides regularization
        epochs: int = 50,
        val_split: float = 0.10,
        val_data: Tuple[np.ndarray, np.ndarray] = None,
        patience: int = 10,
        output_dir_path: str = '../../models/',
        class_weights: Dict[int, float] = None) -> None:
    save_path = os.path.join(
        output_dir_path,
        model_name + '_e{epoch:02d}_vl{val_loss:.2f}_vlacc{val_acc:.4f}.h5'
    )
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=patience,
                       verbose=1)
    cp = ModelCheckpoint(save_path,
                         monitor='val_loss',
                         verbose=1,
                         save_best_only=True)
    classifier.fit(x,
                   y,
                   shuffle=True,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_split=val_split,
                   validation_data=val_data,
                   callbacks=[es, cp],
                   class_weight=class_weights)


def classifier_training_example(mels_path: str) -> None:
    batch_size = 64
    epochs = 50
    model_name = 'class_cnn3'
    patience = 10
    val_split = 0.25

    classifier = build_cnn3_classifier()
    classifier.compile(optimizer='adam',
                       loss='categorical_crossentropy',
                       metrics=['acc'])
    classifier.summary()

    mels_npz = np.load(mels_path)
    samples_x = mels_npz['samples_x']
    samples_y = mels_npz['samples_y']
    log.info(f'Training samples x shape = {samples_x.shape}')
    log.info(f'Training samples y shape = {samples_y.shape}')

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(np.argmax(samples_y, axis=-1)),
        y=np.argmax(samples_y, axis=-1)
    )
    log.info(f'Training samples class weights = {class_weights}')
    class_weights = {idx: w for idx, w in enumerate(class_weights)}

    train_drum_classifier(
        classifier,
        x=samples_x,
        y=samples_y,
        model_name=model_name,
        batch_size=batch_size,
        epochs=epochs,
        val_split=val_split,
        patience=patience,
        class_weights=class_weights)
