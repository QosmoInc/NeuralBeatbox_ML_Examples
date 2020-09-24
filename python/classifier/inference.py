import logging
import os

import librosa as lr
import numpy as np
from tensorflow.keras.models import load_model

from classifier.preprocessing import SR, HOP_LENGTH, N_MELS, N_FFT, \
    get_mel_spec, CLASSES

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def classifier_inference_example(model_name: str,
                                 sample_path: str) -> None:
    sr = SR
    hop_length = HOP_LENGTH
    n_mels = N_MELS
    n_fft = N_FFT
    max_len_samples = 32767  # Ensures output mel shapes are 128 x 128
    normalize_audio = True
    normalize_mel = True

    log.info(f'Sample to be classified: {sample_path}')
    sample_audio, _ = lr.load(sample_path, sr=sr, mono=True)
    sample_mel_spec = get_mel_spec(sample_audio,
                                   sr=sr,
                                   hop_length=hop_length,
                                   n_mels=n_mels,
                                   n_fft=n_fft,
                                   max_len_samples=max_len_samples,
                                   normalize_audio=normalize_audio,
                                   normalize_mel=normalize_mel)

    sample_mel_spec = np.expand_dims(sample_mel_spec, axis=[0, -1])

    log.info(f'Sample audio shape = {sample_audio.shape}')
    log.info(f'Sample mel spec shape = {sample_mel_spec.shape}')

    classifier = load_model(f'../../models/{model_name}')
    prediction = classifier.predict(sample_mel_spec)
    predicted_class = np.argmax(prediction)
    predicted_class_name = CLASSES[predicted_class]

    log.info(f'Classifier predicted class: "{predicted_class_name}" '
             f'with value {prediction[0][predicted_class]:.4f}')
