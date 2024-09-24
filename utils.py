import os

import numpy as np
from scipy.signal import butter, filtfilt


def denoise_find_r_peaks_elgendi(recording, frequency):
    cutoff_frequency = [8, 20]
    sampling_rate = frequency
    order = 3

    filtered_signal, b, a = butterworth_bandpass(recording, cutoff_frequency, sampling_rate, order)

    squared_signal = filtered_signal ** 2
    # squared_signal = recording ** 2

    qrs_window_size = 59
    beat_window_size = 305
    moving_average_qrs = moving_average(squared_signal, qrs_window_size)
    moving_average_beat = moving_average(squared_signal, beat_window_size)

    mean_squared = np.mean(squared_signal)
    beta = 0.08
    alpha = beta * mean_squared
    threshold_1 = moving_average_beat + alpha

    block_demarcation = np.where(moving_average_qrs > threshold_1, 0.1, 0)

    blocks = []
    current_block = []
    for i, value in enumerate(block_demarcation):
        if value == 0.1:
            current_block.append(i)
        else:
            if current_block:
                blocks.append(current_block)
                current_block = []
    if current_block:
        blocks.append(current_block)

    blocks_of_interest = [block for block in blocks if len(block) >= qrs_window_size]

    r_peak_indices = [np.argmax(squared_signal[block]) + block[0] for block in blocks_of_interest]

    return r_peak_indices, filtered_signal


def butterworth_bandpass(data, cutoff_frequency, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = [freq / nyquist for freq in cutoff_frequency]
    b, a = butter(order, normal_cutoff, btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y, b, a


def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


def resample_cycle(cycle, target_length):
    original_indices = np.arange(len(cycle))
    target_indices = np.linspace(0, len(cycle) - 1, target_length)
    resampled_cycle = np.interp(target_indices, original_indices, cycle)
    return resampled_cycle


def load_header(header_file):
    with open(header_file, 'r') as f:
        header = f.read()
    return header


def load_header_with_fallback(header_file):
    try:
        return load_header(header_file)
    except UnicodeDecodeError:
        with open(header_file, 'r', encoding='utf-8') as f:
            header = f.read()
        return header


def find_subfolders(training_folder):
    dataset_paths = []
    for root, dirs, files in os.walk(training_folder):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            dataset_paths.append(subfolder_path)
        break  # Stop os.walk from going into sub-subfolders
    return dataset_paths
