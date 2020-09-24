import logging
import os

from tensorflow.keras.layers import Input, Dense, Flatten, MaxPooling2D, \
    Conv2D, Dropout
from tensorflow.keras.models import Model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def build_cnn3_classifier(
        mel_spec_x: int = 128,
        mel_spec_y: int = 128,
        n_class: int = 9,
        dropout_rate: float = 0.50) -> Model:
    input_img = Input(shape=(mel_spec_x, mel_spec_y, 1))
    x = Conv2D(32,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(input_img)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Conv2D(64,
               (3, 3),
               strides=(1, 1),
               padding='same',
               activation='elu')(x)
    x = MaxPooling2D((4, 4))(x)
    x = Flatten()(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(n_class, activation='softmax')(x)
    model = Model(input_img, x)
    return model
