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
    data_rows = []
    
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
            # print(cycle_durations)
            if all_lead_cycles is None:
                continue

            diagnosis = get_labels(header)
            if len(diagnosis) == 1 and int(diagnosis[0]) in disease_labels:
                cnt += 1

                for cycle, duration in zip(lead1_cycles, cycle_durations):
                    # Create a data row with diagnosis, cycle data, and duration
                    row_data = {'diagnosis': diagnosis, 'cycle_duration': duration}
                    
                    # Add each point in the cycle to the row
                    for j in range(len(cycle)):
                        row_data[f'point_{j + 1}'] = cycle[j]
                    
                    data_rows.append(row_data)

            if max_circle is not None and len(data_rows) >= max_circle:
                break

    df = pd.DataFrame(data_rows)
    print(f"Number of {data_type} records: {cnt}")
    return df, []



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
