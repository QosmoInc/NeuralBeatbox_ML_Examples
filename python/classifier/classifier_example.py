import os

from classifier.inference import classifier_inference_example
from classifier.preprocessing import find_sorted_classifier_samples, \
    find_classifier_samples, calc_mel_spec_example
from classifier.training import classifier_training_example

if __name__ == '__main__':
    samples_root_dir = '../../data/classifier/samples'
    samples_save_path = '../../data/classifier/samples.npz'
    samples_are_sorted = True

    if samples_are_sorted:
        # Use this if samples are sorted into folders like in the example data;
        # folder structure must follow the same naming convention as the example
        find_sorted_classifier_samples(samples_root_dir, samples_save_path)
    else:
        # Use this if you want to crawl a large, unorganized sample library.
        # Assigned labels will be less reliable
        find_classifier_samples(samples_root_dir, samples_save_path)

    # Calculate Mel spectrograms and classes for the found samples
    calc_mel_spec_example(samples_save_path)

    mels_path = '../../data/classifier/' \
                'mels_sr16000_hl256_nm128_nf1024_mls32767_naT_nmT.npz'

    # Train classifier model
    classifier_training_example(mels_path)

    # Change model name to match the best one that was trained
    model_name = 'class_cnn3_e17_vl0.72_vlacc0.7590.h5'

    test_sample_names = ['Acetone Rhythm Ace-MaxV - KICK4.wav',
                         'Alesis D4fx-MaxV - HiHat Open 2.wav',
                         'Boss DR-110-DR-110Clap.wav']

    # Demonstrate classifier model on un-seen samples
    for test_sample_name in test_sample_names:
        test_sample_path = os.path.join(
            '../../data/classifier/drum_singleshots', test_sample_name)

        classifier_inference_example(model_name, test_sample_path)
