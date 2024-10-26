import pandas as pd
import scipy
from scipy.io import loadmat

from helper_code import find_all_challenge_files, get_frequency, get_labels, get_num_samples
from utils import load_header_with_fallback, butterworth_elgendi_rpeak, denoise_find_r_peaks_elgendi
from scipy.signal import resample

import numpy as np
import matplotlib.pyplot as plt
from const import *


def extract_ecg_cycles(recording, frequency, num_samples, target_length=256, cycle_num=5, overlap=3):
    lead = recording[0]  # Choose the first lead here

    if len(lead) != num_samples:
        return None, None

    r_peak_indices, filtered_signal = butterworth_elgendi_rpeak(lead, frequency)

    if r_peak_indices is None or len(r_peak_indices) < cycle_num + 1:
        return None, None

    cycles = []
    durations = []

    # Calculate the step size based on the overlap
    step = cycle_num - overlap
    if step <= 0:
        step = 1  # Ensure at least moving forward by one R-peak to avoid infinite loops

    for index in range(0, len(r_peak_indices) - cycle_num, step):
        # Extract the segment from the current R-peak to the R-peak at the end of the specified cycle number
        start_idx = r_peak_indices[index]
        end_idx = r_peak_indices[index + cycle_num]
        current_ecg = lead[start_idx:end_idx]

        # Resample the current ECG segment to have a uniform length
        current_ecg = resample(current_ecg, target_length * cycle_num)

        
        # t = np.arange(len(current_ecg)) / frequency  # Time vector in seconds

        # plt.figure(figsize=(12, 6))
        # plt.plot(t, current_ecg, color='blue', linewidth=1)
        # # plt.title('ECG Signal')
        # # plt.xlabel('Time (seconds)')
        # # plt.ylabel('Amplitude (mV)')
        # plt.grid(True)
        # plt.show()

        # Compute durations for each cycle in the segment
        cycle_durations = []
        for k in range(cycle_num):
            duration = (r_peak_indices[index + k + 1] - r_peak_indices[index + k]) / frequency
            cycle_durations.append(duration)

        cycles.append(current_ecg)
        durations.append(cycle_durations)

    return cycles, durations


# def load_data(paths, max_circle=None, cycle_num=1, overlap=0):
#     cnt = 0
#     data_rows = []

#     duration_list = []
    
#     for path in paths:
#         header_files, recording_files = find_all_challenge_files(path)
#         length = len(header_files)

#         for i in range(length):
#             mat_file_path = recording_files[i]
#             header_file_path = header_files[i]
#             try:
#                 mat = loadmat(mat_file_path)
#             except (scipy.io.matlab._miobase.MatReadError, FileNotFoundError):
#                 continue
#             header = load_header_with_fallback(header_file_path)
#             frequency = get_frequency(header)
#             num_samples = get_num_samples(header)

#             # Extract ECG cycles and individual cycle durations
#             lead1_cycles, cycle_durations = extract_ecg_cycles(
#                 mat['val'], frequency, num_samples, target_length=256, cycle_num=cycle_num, overlap=overlap
#             )

#             if lead1_cycles is None:
#                 continue

#             diagnosis = get_labels(header)

#             ### Level1
#             if len(diagnosis) == 1 and int(diagnosis[0]) in diagnosis_level_1:
#                 cnt += 1

#                 for cycle, durations in zip(lead1_cycles, cycle_durations):
#                     # Create a data row with diagnosis and durations
#                     row_data = {'diagnosis': 1}

#                     # Add each point in the cycle to the row
#                     for j in range(len(cycle)):
#                         row_data[f'point_{j + 1}'] = cycle[j]

#                     # Add individual cycle durations as separate features
#                     for idx, duration in enumerate(durations):
#                         row_data[f'cycle_duration_{idx + 1}'] = duration
#                         duration_list.append(duration)

#                     data_rows.append(row_data)

#             ### Level2
#             if len(diagnosis) == 1 and int(diagnosis[0]) in diagnosis_level_2:
#                 cnt += 1

#                 for cycle, durations in zip(lead1_cycles, cycle_durations):
#                     # Create a data row with diagnosis and durations
#                     row_data = {'diagnosis': 2}

#                     # Add each point in the cycle to the row
#                     for j in range(len(cycle)):
#                         row_data[f'point_{j + 1}'] = cycle[j]

#                     # Add individual cycle durations as separate features
#                     for idx, duration in enumerate(durations):
#                         row_data[f'cycle_duration_{idx + 1}'] = duration
#                         duration_list.append(duration)

#                     data_rows.append(row_data)

#             ### Level3
#             if len(diagnosis) == 1 and int(diagnosis[0]) in diagnosis_level_3:
#                 cnt += 1

#                 for cycle, durations in zip(lead1_cycles, cycle_durations):
#                     # Create a data row with diagnosis and durations
#                     row_data = {'diagnosis': 3}

#                     # Add each point in the cycle to the row
#                     for j in range(len(cycle)):
#                         row_data[f'point_{j + 1}'] = cycle[j]

#                     # Add individual cycle durations as separate features
#                     for idx, duration in enumerate(durations):
#                         row_data[f'cycle_duration_{idx + 1}'] = duration
#                         duration_list.append(duration)

#                     data_rows.append(row_data)

#             ### Level4
#             if len(diagnosis) == 1 and int(diagnosis[0]) in diagnosis_level_4:
#                 cnt += 1

#                 for cycle, durations in zip(lead1_cycles, cycle_durations):
#                     # Create a data row with diagnosis and durations
#                     row_data = {'diagnosis': 4}

#                     # Add each point in the cycle to the row
#                     for j in range(len(cycle)):
#                         row_data[f'point_{j + 1}'] = cycle[j]

#                     # Add individual cycle durations as separate features
#                     for idx, duration in enumerate(durations):
#                         row_data[f'cycle_duration_{idx + 1}'] = duration
#                         duration_list.append(duration)

#                     data_rows.append(row_data)

#             ### Level5
#             if len(diagnosis) == 1 and int(diagnosis[0]) in diagnosis_level_5:
#                 cnt += 1

#                 for cycle, durations in zip(lead1_cycles, cycle_durations):
#                     # Create a data row with diagnosis and durations
#                     row_data = {'diagnosis': 5}

#                     # Add each point in the cycle to the row
#                     for j in range(len(cycle)):
#                         row_data[f'point_{j + 1}'] = cycle[j]

#                     # Add individual cycle durations as separate features
#                     for idx, duration in enumerate(durations):
#                         row_data[f'cycle_duration_{idx + 1}'] = duration
#                         duration_list.append(duration)

#                     data_rows.append(row_data)

#             if max_circle is not None and len(data_rows) >= max_circle:
#                 break

#     df = pd.DataFrame(data_rows)
#     print(f"Number of records: {cnt}")
#     return df, duration_list

import numpy as np
import scipy.io

def load_data(paths, max_circle=None, cycle_num=1, overlap=0):
    cnt = 0
    data_rows = []
    duration_list = []

    for path in paths:
        header_files, recording_files = find_all_challenge_files(path)
        length = len(header_files)

        for i in range(length):
            mat_file_path = recording_files[i]
            header_file_path = header_files[i]

            try:
                mat = loadmat(mat_file_path)
            except (scipy.io.matlab._miobase.MatReadError, FileNotFoundError):
                continue

            header = load_header_with_fallback(header_file_path)
            frequency = get_frequency(header)
            num_samples = get_num_samples(header)

            # Extract ECG cycles and individual cycle durations
            lead1_cycles, cycle_durations = extract_ecg_cycles(
                mat['val'], frequency, num_samples, target_length=256, cycle_num=cycle_num, overlap=overlap
            )

            if lead1_cycles is None:
                continue

            diagnosis = get_labels(header)
            if len(diagnosis) == 1:
                diag_code = int(diagnosis[0])
                
                if diag_code in diagnosis_level_1:
                    diag_label = 1
                elif diag_code in diagnosis_level_2:
                    diag_label = 2
                elif diag_code in diagnosis_level_3:
                    diag_label = 3
                elif diag_code in diagnosis_level_4:
                    diag_label = 4
                elif diag_code in diagnosis_level_5:
                    diag_label = 5
                else:
                    continue

                cnt += 1

                # Prepare rows for each cycle
                for cycle, durations in zip(lead1_cycles, cycle_durations):
                    row_data = [diag_label] + list(cycle) + list(durations)
                    data_rows.append(row_data)

                    # Add individual cycle durations to the list
                    duration_list.extend(durations)

            if max_circle is not None and len(data_rows) >= max_circle:
                break

    # Convert the list of lists to a NumPy array for faster dataframe creation
    data_array = np.array(data_rows)

    # Create a DataFrame with column names
    num_points = len(lead1_cycles[0]) if lead1_cycles else 0
    num_durations = len(cycle_durations[0]) if cycle_durations else 0

    columns = ['diagnosis'] + [f'point_{i + 1}' for i in range(num_points)] + [f'cycle_duration_{i + 1}' for i in range(num_durations)]
    df = pd.DataFrame(data_array, columns=columns)

    print(f"Number of records: {cnt}")
    return df, duration_list





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
