import os

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def moving_window_integration(signal, window_size):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


# Dynamic thresholding to detect QRS complexes
def dynamic_thresholding(signal, threshold_factor=0.5):
    threshold = np.mean(signal) + threshold_factor * np.std(signal)
    qrs_points = np.where(signal > threshold)[0]
    return qrs_points, threshold


def calculate_gap_threshold(frequency, min_bpm=40):
    min_rr_interval = 60 / min_bpm  # in seconds
    gap_threshold = int(0.5 * min_rr_interval * frequency)  # half of the minimum RR interval in samples
    return gap_threshold


def get_top_indices_per_group(indices, ecg_data, gap_threshold):
    top_indices = []
    start_idx = 0

    for i in range(1, len(indices)):
        # If the gap between two consecutive indices is greater than the threshold
        if indices[i] - indices[i - 1] > gap_threshold:
            # Find the index with the maximum value in the current group
            group = indices[start_idx:i]
            if len(group) > 0:
                top_1_in_group = group[np.argmax(ecg_data[group])]
                top_indices.append(top_1_in_group)
            start_idx = i  # Update the start index for the next group

    # Handle the last group
    group = indices[start_idx:]
    if len(group) > 0:
        top_1_in_group = group[np.argmax(ecg_data[group])]
        top_indices.append(top_1_in_group)

    return top_indices


def find_r_peaks(indices, ecg_signal):
    """
    Finds the R-peaks in the ECG signal based on grouped indices.

    Parameters:
    - indices: List of indices corresponding to QRS complexes in the ECG signal.
    - ecg_signal: The ECG signal array.

    Returns:
    - r_peaks: List of indices corresponding to the R-peaks in the ECG signal.
    """

    # if not indices:
    #     return []

    # Sort indices to ensure proper grouping
    indices = sorted(indices)
    
    # Group indices into consecutive sequences
    groups = []
    current_group = [indices[0]]

    for idx in indices[1:]:
        if idx == current_group[-1] + 1:
            # Consecutive index, add to current group
            current_group.append(idx)
        else:
            # Non-consecutive index, start a new group
            groups.append(current_group)
            current_group = [idx]
    # Append the last group
    groups.append(current_group)

    # Find R-peak in each group (maximum amplitude in ecg_signal)
    r_peaks = []
    for group in groups:
        group_indices = group
        group_values = ecg_signal[group_indices]
        # Find index of maximum value within the group
        max_idx_in_group = group_indices[np.argmax(group_values)]
        r_peaks.append(max_idx_in_group)

    return r_peaks


def butterworth_elgendi_rpeak(recording, frequency):

    raw_ecg_signal = recording

    # Apply bandpass filter
    cutoff_frequency_qrs = [8, 20]  # Hz
    filtered_ecg_signal, b, a = butterworth_bandpass(raw_ecg_signal, cutoff_frequency_qrs, frequency, order=3)

    # Square the filtered signal
    squared_signal_qrs = filtered_ecg_signal ** 2

    # Apply moving window integration
    window_size_qrs = int(0.150 * frequency)
    mwi_signal_qrs = moving_window_integration(squared_signal_qrs, window_size_qrs)

    # Dynamic thresholding
    qrs_points, threshold_qrs = dynamic_thresholding(mwi_signal_qrs, 0.1)

    if len(qrs_points) == 0:
        return None, None

    # Calculate dynamic gap threshold
    # gap_threshold = calculate_gap_threshold(frequency, min_bpm=40)

    # Get R-peaks using the modified function
    # r_peaks = get_top_indices_per_group(qrs_points, filtered_ecg_signal, 10)

    r_peaks = find_r_peaks(qrs_points, filtered_ecg_signal)


    # t = np.arange(len(recording)) / frequency

    # # Visualize the process and detected QRS, T, and P points
    # plt.figure(figsize=(12, 8))

    # # QRS Detection Steps
    # plt.subplot(5, 1, 1)
    # plt.plot(t, raw_ecg_signal, label='Raw ECG Signal')
    # plt.title("Raw ECG Signal")
    # plt.legend()


    # plt.subplot(5, 1, 2)
    # plt.plot(t, filtered_ecg_signal, label='Butterworth Filtered ECG Signal (QRS Detection)')
    # plt.title("Butterworth Filtered ECG Signal for QRS Detection")
    # plt.legend()

    # # Squared Signal
    # plt.subplot(5, 1, 3)
    # plt.plot(t, squared_signal_qrs, label='Squared Signal')
    # plt.title("Squared ECG Signal")
    # plt.legend()

    # plt.subplot(5, 1, 4)
    # plt.plot(t, mwi_signal_qrs, label='MWI Signal')
    # plt.axhline(y=threshold_qrs, color='r', linestyle='--', label='Threshold')
    # plt.plot(t[qrs_points], mwi_signal_qrs[qrs_points], 'ro', label='Detected QRS Points')
    # plt.title("MWI Signal with Detected QRS Points")
    # plt.legend()

    # plt.subplot(5, 1, 5)
    # plt.plot(t, raw_ecg_signal, label='Raw ECG Signal')
    # plt.plot(t[r_peaks], raw_ecg_signal[r_peaks], 'ro', label='Detected QRS Points')
    # plt.title("Detected QRS Points on ECG Signal")
    # plt.legend()


    # plt.tight_layout()

    # plt.show()

    return r_peaks, filtered_ecg_signal

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
