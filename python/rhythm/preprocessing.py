import logging
import os
from typing import Dict, List

import joblib
import numpy as np
import xxhash
from pretty_midi import PrettyMIDI, Instrument
from scipy.stats import zscore
from tqdm import tqdm

from rhythm.midi_maps import CHRISTHETREE_MIDI_MAP_8

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

DEFAULT_MIDI_MAP = CHRISTHETREE_MIDI_MAP_8

N_BARS = 2
Q_NOTE_RES = 4
LEN_SEQ = N_BARS * 4 * Q_NOTE_RES

N_DRUMS = 8
MAX_CONT_RESTS = 4 * Q_NOTE_RES  # 1 bar of rests
MIN_ONSETS = 8


def _calc_drum_matrices(
        pm: PrettyMIDI,
        instrument: Instrument,
        midi_total_notes: int,
        matrix_total_notes: int,
        ticks_per_note: float,
        hop_size: int,
        n_drums: int,
        midi_drum_map: Dict[int, int],
        min_onsets: int
) -> (List[np.ndarray], List[np.ndarray]):
    onset_matrices = []
    vel_matrices = []

    # Create one matrix for the entire MIDI to prevent mem-alloc ops
    midi_onset_matrix = np.zeros((midi_total_notes, n_drums), dtype=np.int32)
    midi_vel_matrix = np.zeros((midi_total_notes, n_drums), dtype=np.int32)

    for note in instrument.notes:
        if note.pitch in midi_drum_map and note.velocity > 0:
            note_start_ticks = pm.time_to_tick(note.start)
            note_idx = int((note_start_ticks / ticks_per_note) + 0.5)
            drum_idx = midi_drum_map[note.pitch]
            vel = note.velocity

            # Ignore notes quantized forward past last beat of MIDI file
            if note_idx < midi_total_notes:
                midi_onset_matrix[note_idx, drum_idx] = 1

                prev_vel = midi_vel_matrix[note_idx, drum_idx]
                # If multiple hits exist, use the louder one
                midi_vel_matrix[note_idx, drum_idx] = max(prev_vel, vel)
        else:
            log.debug(
                f'MIDI note pitch of {note.pitch} not found or vel is '
                f'equal to 0. Vel = {note.velocity}')

    # Create data points via a single pass over entire MIDI matrix
    for start_note in range(
            0, midi_total_notes - matrix_total_notes + 1, hop_size):
        end_note = start_note + matrix_total_notes
        sub_onset_matrix = midi_onset_matrix[start_note:end_note, :]
        num_of_onsets = np.sum(sub_onset_matrix)

        if num_of_onsets >= min_onsets:
            onset_matrices.append(sub_onset_matrix)

            # Avoid unnecessary ndarray creation
            sub_vel_matrix = midi_vel_matrix[start_note:end_note, :]
            vel_matrices.append(sub_vel_matrix)

    return onset_matrices, vel_matrices


def _calc_matrices(
        midi_file_path: str,
        n_bars: int = N_BARS,
        q_note_res: int = Q_NOTE_RES,
        n_drums: int = N_DRUMS,
        midi_drum_map: Dict[int, int] = DEFAULT_MIDI_MAP,
        min_onsets: int = MIN_ONSETS,
        no_time_sig_means_44: bool = False
) -> (List[np.ndarray], List[np.ndarray]):
    matrix_total_notes = 4 * q_note_res * n_bars
    hop_size = 4 * q_note_res
    onset_matrices = []
    vel_matrices = []

    try:
        pm = PrettyMIDI(midi_file_path)
    except:
        log.error(f'Failed to load MIDI file. {midi_file_path}')
        return onset_matrices, vel_matrices

    if not pm.time_signature_changes and not no_time_sig_means_44:
        log.debug(f'MIDI file contains no time signatures. {midi_file_path}')
        return onset_matrices, vel_matrices

    if not pm.time_signature_changes:
        time_sigs = [(4, 4)]
    else:
        time_sigs = [(ts.numerator, ts.denominator)
                     for ts in pm.time_signature_changes]
    ts_nums, ts_denoms = zip(*time_sigs)

    # TODO(christhetree): genericize
    if not all([num == 4 for num in ts_nums]) \
            or not all([denom == 4 for denom in ts_denoms]):
        log.debug(
            f'MIDI file is not entirely 4/4. {midi_file_path}')
        return onset_matrices, vel_matrices

    if not pm.instruments:
        log.debug(
            f'MIDI file does not contain any instruments. {midi_file_path}')
        return onset_matrices, vel_matrices

    q_note_times = pm.get_beats()
    num_of_q_notes = len(q_note_times)
    midi_total_notes = num_of_q_notes * q_note_res
    ticks_per_note = pm.resolution / q_note_res

    if midi_total_notes < matrix_total_notes:
        log.debug(f'MIDI file is too short. {midi_file_path}')
        return onset_matrices, vel_matrices

    for instrument in pm.instruments:
        if instrument.is_drum:
            onset_m_s, vel_m_s = _calc_drum_matrices(pm,
                                                     instrument,
                                                     midi_total_notes,
                                                     matrix_total_notes,
                                                     ticks_per_note,
                                                     hop_size,
                                                     n_drums,
                                                     midi_drum_map,
                                                     min_onsets)
            onset_matrices.extend(onset_m_s)
            vel_matrices.extend(vel_m_s)

    assert len(onset_matrices) == len(vel_matrices)
    return onset_matrices, vel_matrices


def _deduplicate_drum_matrices(
        onset_matrices: List[np.ndarray],
        vel_matrices: List[np.ndarray]
) -> (List[np.ndarray], List[np.ndarray]):
    assert len(onset_matrices) == len(vel_matrices)

    vel_hashes = set()
    hasher = xxhash.xxh64()
    unique_onset_matrices = []
    unique_vel_matrices = []

    log.info(f'Beginning deduplication of {len(onset_matrices)} matrices.')
    for onset_m, vel_m in tqdm(zip(onset_matrices, vel_matrices)):
        vel_string = np.array_str(vel_m)

        hasher.update(vel_string)
        vel_hash = hasher.hexdigest()
        hasher.reset()

        if vel_hash not in vel_hashes:
            vel_hashes.add(vel_hash)
            unique_onset_matrices.append(onset_m)
            unique_vel_matrices.append(vel_m)

    assert len(unique_onset_matrices) == len(unique_vel_matrices)
    log.info(f'Unique matrices length = {len(unique_onset_matrices)}')
    return unique_onset_matrices, unique_vel_matrices


def get_drum_matrices(
        root_dir: str,
        save_path: str,
        file_ending: str = '.mid',
        n_bars: int = N_BARS,
        q_note_res: int = Q_NOTE_RES,
        n_drums: int = N_DRUMS,
        midi_drum_map: Dict[int, int] = DEFAULT_MIDI_MAP,
        min_onsets: int = MIN_ONSETS,
        n_jobs: int = -1
) -> (np.ndarray, np.ndarray):
    file_paths = []
    for root, dirs, files in tqdm(os.walk(root_dir)):
        for f in files:
            if f.endswith(file_ending):
                file_paths.append(os.path.join(root, f))

    log.info(f'Found {len(file_paths)} files ending with {file_ending}')

    if n_jobs == 1:
        all_matrices = []
        for path in tqdm(file_paths):
            onset_matrices, vel_matrices = _calc_matrices(path,
                                                          n_bars,
                                                          q_note_res,
                                                          n_drums,
                                                          midi_drum_map,
                                                          min_onsets)
            all_matrices.append((onset_matrices, vel_matrices))
    else:
        all_matrices = joblib.Parallel(n_jobs=-1, verbose=5)(
            joblib.delayed(_calc_matrices)(path,
                                           n_bars,
                                           q_note_res,
                                           n_drums,
                                           midi_drum_map,
                                           min_onsets)
            for path in file_paths)

    all_onset_matrices, all_vel_matrices = zip(*all_matrices)

    flat_onset_matrices = [m for sublist in all_onset_matrices for m in sublist]
    log.info(f'Onset matrices length = {len(flat_onset_matrices)}')

    flat_vel_matrices = [m for sublist in all_vel_matrices for m in sublist]
    log.info(f'Velocity matrices length = {len(flat_vel_matrices)}')

    unique_onset_matrices, unique_vel_matrices = _deduplicate_drum_matrices(
        flat_onset_matrices, flat_vel_matrices)

    np_onset_matrices = np.array(unique_onset_matrices, dtype=np.float32)
    np_vel_matrices = np.array(unique_vel_matrices, dtype=np.float32) / 127

    np.savez(save_path,
             drum_onset_matrices=np_onset_matrices,
             drum_vel_matrices=np_vel_matrices)

    return np_onset_matrices, np_vel_matrices


def calc_conditioning_data(
        onset_x: np.ndarray,
        binary_only: bool = False,
        use_zscore: bool = True
) -> (np.ndarray, np.ndarray, np.ndarray):
    no_of_beats = onset_x.shape[1]
    binary_drums = np.where(onset_x > 0.0, 1.0, 0.0)
    drums_sums = np.sum(binary_drums, axis=1)

    if binary_only:
        cond = np.minimum(drums_sums, 1.0)
    else:
        cond = drums_sums / no_of_beats

    cond_mean = np.mean(cond, axis=0)
    cond_std = np.std(cond, axis=0)

    if use_zscore:
        cond = zscore(cond)

    return cond, cond_mean, cond_std


def rhythm_dataset_example(root_dir: str,
                           save_dir: str) -> None:
    save_path = os.path.join(
        save_dir,
        f'groove_v2_nob{N_BARS}_qnr{Q_NOTE_RES}_nod{N_DRUMS}_mo{MIN_ONSETS}.npz'
    )

    file_ending = 'beat_4-4.mid'  # Required for groove dataset
    # file_ending = '.mid'

    midi_map = CHRISTHETREE_MIDI_MAP_8  # Map to 8 different drums

    get_drum_matrices(root_dir,
                      save_path,
                      file_ending,
                      midi_drum_map=midi_map,
                      n_jobs=-1)
