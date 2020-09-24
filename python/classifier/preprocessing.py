import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import librosa as lr
import numpy as np
import xxhash as xxhash
from audioread import NoBackendError
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

CLASSES = [
    'kick',
    'snare',
    'hihat_closed',
    'hihat_open',
    'tom_low',
    'tom_mid',
    'tom_high',
    'clap',
    'rim'
]

CLASS_SEARCH_WORDS = {
    0: [['kick']],
    1: [['snare']],
    2: [['hihat', 'hi-hat', 'hat'], ['open']],
    3: [['hihat', 'hi-hat', 'hat'], ['close']],
    4: [['tom'], ['low', 'floor', 'lo_']],
    5: [['tom'], ['mid']],
    6: [['tom'], ['high', 'hi_']],
    7: [['clap']],
    8: [['_rim', 'rim_']],
}

FILE_ENDINGS = ('.wav', '.aif', '.ogg', '.flac')
SAMPLE_MAX_SIZE = 2000000
SAMPLE_MAX_DUR = 5.0

SR = 16000
HOP_LENGTH = 256
N_MELS = 128
N_FFT = 1024
MEL_MAX_DUR = 3.0


def _get_eligible_classes(
        path: str,
        class_search_words: Dict[int, List[List[str]]]
) -> List[int]:
    eligible_classes = []
    for class_key, search_words in class_search_words.items():
        if all([any([w in path for w in or_words])
                for or_words in search_words]):
            eligible_classes.append(class_key)

    return eligible_classes


def _save_samples(path_s_classes: List[Tuple[str, int]],
                  save_path: str,
                  max_dur: float,
                  sr: int) -> None:
    hasher = xxhash.xxh64()
    sample_hashes = set()

    if max_dur:
        max_load_duration = max_dur + 0.01  # Increase speed of loading
    else:
        max_load_duration = None

    audio_matrices = []
    assigned_classes = []
    paths = []
    np.random.shuffle(path_s_classes)

    for path, assigned_class in tqdm(path_s_classes):
        try:
            audio, _ = lr.load(
                path, sr=sr, mono=True, duration=max_load_duration)
        except (EOFError, NoBackendError, RuntimeError):
            log.error(f'Failed to load file: {path}')
            continue

        if max_dur and lr.core.get_duration(audio, sr) > max_dur:
            log.info(f'Sample is too long: {path}')
            continue

        hasher.update(audio.tostring())
        sample_hash = hasher.hexdigest()
        hasher.reset()

        if sample_hash in sample_hashes:
            log.info(f'Duplicate sample found: {path}')
            continue
        else:
            sample_hashes.add(sample_hash)

        audio_matrices.append(audio)
        assigned_classes.append(assigned_class)
        paths.append(path)

    log.info(f'Length of audio matrices = {len(audio_matrices)}')
    log.info(f'Length of assigned classes = {len(assigned_classes)}')
    log.info(f'Length of paths = {len(paths)}')

    np.savez(save_path,
             audio_matrices=audio_matrices,
             assigned_classes=assigned_classes,
             paths=paths)


def find_classifier_samples(
        root_dir: str,
        save_path: str,
        file_endings: Tuple = FILE_ENDINGS,
        class_search_words: Dict[int, List[List[str]]] = CLASS_SEARCH_WORDS,
        class_priority: List[int] = None,
        max_size: int = SAMPLE_MAX_SIZE,
        max_dur: float = SAMPLE_MAX_DUR,
        sr: int = SR
) -> None:
    def process_path(path: str) -> str:
        proc_path = path.lower()
        # Apparently this is faster than using a regex
        proc_path = proc_path.replace(' ', '_') \
            .replace('-', '_') \
            .replace('.', '_') \
            .replace('/', '_') \
            .replace('\\', '_')
        return proc_path

    file_paths = []

    for root, dirs, files in tqdm(os.walk(root_dir)):
        for f in files:
            if f.endswith(file_endings) and not f.startswith('._'):
                path = os.path.join(root, f)
                file_size = os.path.getsize(path)
                if file_size <= max_size:
                    file_paths.append(path)

    log.info(
        f'Found {len(file_paths)} small enough {file_endings} files')

    if not file_paths:
        return

    path_m_classes = []
    m_class_dist = defaultdict(int)
    for path in tqdm(file_paths):
        proc_path = process_path(path)
        eligible_classes = _get_eligible_classes(proc_path, class_search_words)
        for c in eligible_classes:
            m_class_dist[c] += 1
        path_m_classes.append(eligible_classes)

    log.info(f'Num of classes assigned to paths = {sum(m_class_dist.values())}')
    for c, freq in sorted(m_class_dist.items()):
        log.info(f'Class {c} num of paths = {freq}')

    if class_priority is None:
        sorted_dist = sorted(list(m_class_dist.items()), key=lambda x: x[1])
        sorted_dist, _ = zip(*sorted_dist)
        sorted_dist = sorted(enumerate(sorted_dist), key=lambda x: x[1])
        class_priority, _ = zip(*sorted_dist)
        class_priority = list(class_priority)

    log.info(f'Class priority = {class_priority}')

    s_class_dist = defaultdict(int)
    path_s_classes = []
    for path, eligible_classes in zip(file_paths, path_m_classes):
        if eligible_classes:
            priority_sorted = sorted(
                eligible_classes, key=lambda c: class_priority[c])
            assigned_class = priority_sorted[0]
            s_class_dist[assigned_class] += 1
            path_s_classes.append((path, assigned_class))

    log.info(f'Num of unique class path combos = {sum(s_class_dist.values())}')
    for c, freq in sorted(s_class_dist.items()):
        log.info(f'Class {c} num of unique paths = {freq}')
    log.info(f'Length of class path combos = {len(path_s_classes)}')

    _save_samples(path_s_classes, save_path, max_dur, sr)


def find_sorted_classifier_samples(
        root_dir: str,
        save_path: str,
        file_endings: Tuple = FILE_ENDINGS,
        max_size: int = SAMPLE_MAX_SIZE,
        max_dur: Optional[float] = SAMPLE_MAX_DUR,
        sr: int = SR
) -> None:
    s_class_dist = defaultdict(int)
    path_s_classes = []
    for dir in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir)
        # Prevent hidden files etc. from being counted as classes
        if not dir.startswith('.') and os.path.isdir(dir_path):
            try:
                assigned_class = int(dir[0])
            except ValueError:
                log.error(f'{dir_path} is formatted incorrectly')
                return

            for root, dirs, files in tqdm(os.walk(dir_path)):
                for f in files:
                    if f.endswith(file_endings) and not f.startswith('._'):
                        path = os.path.join(root, f)
                        file_size = os.path.getsize(path)
                        if file_size <= max_size:
                            path_s_classes.append((path, assigned_class))
                            s_class_dist[assigned_class] += 1

    log.info(f'Found {len(path_s_classes)} small enough {file_endings} files')

    if not path_s_classes:
        return

    log.info(f'Num of unique class path combos = {sum(s_class_dist.values())}')
    for c, freq in sorted(s_class_dist.items()):
        log.info(f'Class {c} num of unique paths = {freq}')

    _save_samples(path_s_classes, save_path, max_dur, sr)


def get_mel_spec(audio: np.ndarray,
                 sr: int = SR,
                 hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 max_len_samples: int = None,
                 max_dur: float = MEL_MAX_DUR,
                 normalize_audio: bool = True,
                 normalize_mel: bool = True) -> np.ndarray:
    if max_len_samples is None:
        max_len_samples = int(max_dur * sr)

    if normalize_audio:
        audio = lr.util.normalize(audio)

    audio = audio[:max_len_samples]
    audio_length = audio.shape[0]
    if audio_length < max_len_samples:
        audio = np.concatenate(
            [audio, np.zeros(max_len_samples - audio_length)])

    mel_spec = lr.feature.melspectrogram(
        audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = lr.power_to_db(mel_spec, ref=1.0)

    if normalize_mel:
        # Axis must be None to normalize across both dimensions at once
        mel_spec = lr.util.normalize(mel_spec, axis=None)

    return mel_spec


def calc_mel_spec_example(samples_path: str) -> None:
    samples = np.load(samples_path, allow_pickle=True)
    audio_matrices = samples['audio_matrices']
    assigned_classes = samples['assigned_classes']

    sr = SR
    hop_length = HOP_LENGTH
    n_mels = N_MELS
    n_fft = N_FFT
    max_len_samples = 32767  # Ensures output mel shapes are 128 x 128
    normalize_audio = True
    normalize_mel = True

    mel_specs = []

    for audio in tqdm(audio_matrices):
        mel_spec = get_mel_spec(
            audio,
            sr=sr,
            hop_length=hop_length,
            n_mels=n_mels,
            n_fft=n_fft,
            max_len_samples=max_len_samples,
            normalize_audio=normalize_audio,
            normalize_mel=normalize_mel
        )

        mel_specs.append(mel_spec)

    samples_x = np.expand_dims(np.array(mel_specs), axis=-1)
    log.info(f'samples_x shape = {samples_x.shape}')
    samples_y = to_categorical(assigned_classes, 9)
    log.info(f'samples_y shape = {samples_y.shape}')

    save_name = f'mels_sr{sr}_hl{hop_length}_nm{n_mels}' \
                f'_nf{n_fft}_mls{max_len_samples}_na{str(normalize_audio)[0]}' \
                f'_nm{str(normalize_mel)[0]}.npz'
    np.savez(os.path.join('../../data/classifier', save_name),
             samples_x=samples_x,
             samples_y=samples_y)
