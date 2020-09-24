from rhythm.preprocessing import rhythm_dataset_example
from rhythm.training import rhythm_vae_training_example

if __name__ == '__main__':
    midi_root_dir = '../../data/rhythm/groove'
    midi_save_dir = '../../data/rhythm'

    # Create training dataset
    rhythm_dataset_example(midi_root_dir, midi_save_dir)

    # Multiple training data files will be combined into one
    training_data_paths = ['../../data/rhythm/groove_v2_nob2_qnr4_nod8_mo8.npz']

    # Train rhythm VAE
    rhythm_vae_training_example(training_data_paths)

    # For inference examples, please see render_audio.ipynb
