import logging
import os
from typing import List

import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note

from rhythm.midi_maps import CHRISTHETREE_PLAYBACK_MIDI_MAP_8
from rhythm.preprocessing import Q_NOTE_RES

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


def _add_note(pm_inst: Instrument,
              pitch: int,
              vel: int,
              note_start_s: float,
              note_end_s: float) -> None:
    note = Note(vel, pitch, note_start_s, note_end_s)
    pm_inst.notes.append(note)


def create_drum_pretty_midi(
        onset_matrix: np.ndarray,
        vel_matrix: np.ndarray,
        q_note_res: int = Q_NOTE_RES,
        bpm: float = 120.0,
        playback_midi_map: List[int] = CHRISTHETREE_PLAYBACK_MIDI_MAP_8
) -> PrettyMIDI:
    pm = PrettyMIDI(initial_tempo=bpm)

    res = (60.0 / bpm) / q_note_res
    drum_inst = Instrument(0, is_drum=True)
    n_beats = vel_matrix.shape[0]

    for beat_idx in range(n_beats):
        onset_vals = onset_matrix[beat_idx, :]
        onsets = np.nonzero(onset_vals >= 0.5)[0]
        vels = vel_matrix[beat_idx, :]

        for onset in onsets:
            pitch = playback_midi_map[onset]
            vel = int((vels[onset] * 127) + 0.5)
            _add_note(drum_inst,
                      pitch,
                      vel,
                      res * beat_idx,
                      (res * beat_idx) + 0.2)

    pm.instruments.append(drum_inst)
    return pm
