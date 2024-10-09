import pandas as pd
import scipy
from scipy.io import loadmat

from helper_code import find_all_challenge_files, get_frequency, get_labels, get_num_samples
from utils import load_header_with_fallback, butterworth_elgendi_rpeak, denoise_find_r_peaks_elgendi
from scipy.signal import resample

import numpy as np
import matplotlib.pyplot as plt

fixed_length = 5000  # Define a fixed length for the ECG segments
cycle_length = 256  # Define the length of the cycles


def extract_ecg_cycles(recording, frequency, num_samples, target_length=256, cycle_num=5, overlap=3):

    cycles_matrix = []
    durations_matrix = []

    for lead in recording:
        if len(lead) != num_samples:
            return None, None

        r_peak_indices, filtered_signal = butterworth_elgendi_rpeak(lead, frequency)

        if r_peak_indices is None:
            return None, None
        
        peak_num = cycle_num + 1
        cycles = []
        durations = []

        for index in range(len(r_peak_indices) - peak_num + 1):
            # Extract the segment from the current R-peak to the R-peak at the end of the specified cycle number
            start_idx = r_peak_indices[index]
            end_idx = r_peak_indices[index + peak_num - 1]
            current_ecg = lead[start_idx:end_idx]

            # Calculate duration for the cycle in seconds
            duration = (end_idx - start_idx) / frequency
            durations.append(duration)

            # Resample the current ECG segment to have a uniform length
            current_ecg = resample(current_ecg, target_length * (peak_num - 1))
            cycles.append(current_ecg)

        cycles_matrix.append(cycles)
        durations_matrix.append(durations)
    

    return cycles_matrix, durations_matrix


def load_data(paths, disease_labels=None, data_type="normal", max_circle=None, cycle_num=1, overlap=0):
    cnt = 0
    matrices = []
    
    for path in paths:
        header_files, recording_files = find_all_challenge_files(path)
        length = len(header_files)

        for i in range(length):
            data_matrix = []

            # Get mat and header files
            mat_file_path = recording_files[i]
            header_file_path = header_files[i]

            # Load mat file and header file
            try:
                mat = loadmat(mat_file_path)
            except (scipy.io.matlab._miobase.MatReadError, FileNotFoundError):
                continue
            header = load_header_with_fallback(header_file_path)
            frequency = get_frequency(header)
            num_samples = get_num_samples(header)

            all_lead_cycles, cycle_durations = extract_ecg_cycles(mat['val'], frequency, num_samples, cycle_length, cycle_num, overlap)
            if all_lead_cycles is None:
                continue

            diagnosis = get_labels(header)
            if len(diagnosis) == 1 and int(diagnosis[0]) in disease_labels:
                cnt += 1
                    
                for lead_cycles in all_lead_cycles:
                    lead = []
                    for cycle, duration in zip(lead_cycles, cycle_durations):
                        # Create a data row with cycle data, and duration
                        row_data = {'diagnosis': diagnosis, 'cycle_duration': duration}
                        
                        # Add each point in the cycle to the row
                        for j in range(len(cycle)):
                            row_data[f'point_{j + 1}'] = cycle[j]
                        
                        lead.append(row_data)
                    data_matrix.append(lead)
                    
                matrices.append(data_matrix)

            if max_circle is not None and len(matrices) >= max_circle:
                break

    print(f"Number of {data_type} records: {cnt}")
    return matrices, []

def pad_data(matrices):
    """
    Takes list of matrices and pads them
    Separates labels from matrices, maintaining matching indices
    Returns:
        padded matrices
        corresponding labels
    """

    labels = []
    padded_matrices = []
    padded_durations = []

    for matrix in matrices:
        # Store diagnosis since it's the same for the whole matrix
        diagnosis = matrix[0][0]['diagnosis']
        labels.append(diagnosis)

        # Find max number of cycles in any particular lead
        max_lead_len = max([len(lead) for lead in matrix])

        # Pad leads with extra cycles of 0 values
        padded_leads = []
        padded_lead_durations = []

        for lead in matrix:
            lead_array = []
            lead_duration_array = []

            for cycle in lead:
                cycle_array = []
                for i in range(2, 258):
                    cycle_array.append(cycle[f"point_{i}"])
                lead_array.append(np.array(cycle_array))

                # Extract duration
                lead_duration_array.append(cycle['cycle_duration'])

            # Add padding
            diff = max_lead_len - len(lead)
            for i in range(diff):
                padding_cycle = np.zeros(256)
                lead_array.append(padding_cycle)
                # Duration padding
                lead_duration_array.append(0.0)

            padded_leads.append(np.array(lead_array))
            padded_lead_durations.append(np.array(lead_duration_array))


        # Stack the padded leads to create a padded matrix
        padded_matrix = np.stack(padded_leads)
        padded_matrices.append(padded_matrix)

        padded_durations.append(np.stack(padded_lead_durations))

    return padded_matrices, labels, padded_durations


if __name__ == "__main__":
    from const import *

    dataset_paths = [r"..\..\physionet.org\files\challenge-2021\1.0.3\training\chapman_shaoxing"]
    data_test_1, dataset_test_1 = load_data(dataset_paths, data_type="normal", max_circle=10)
    data_test_2, dataset_test_2 = load_data(dataset_paths, diagnosis_level_5, data_type="abnormal", max_circle=10)
    data_test_3, dataset_test_3 = load_data(dataset_paths, diagnosis_level_5, data_type="single_abnormal", max_circle=10)
    print(data_test_1.head())
    print(dataset_test_1)
    print(data_test_2.head())
    print(dataset_test_2)
    print(data_test_3.head())
    print(dataset_test_3)
