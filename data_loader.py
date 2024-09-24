import pandas as pd
import scipy
from scipy.io import loadmat
import numpy as np
import torch

from helper_code import find_all_challenge_files, get_frequency, get_labels
from utils import resample_cycle, load_header_with_fallback, denoise_find_r_peaks_elgendi

fixed_length = 5000  # Define a fixed length for the ECG segments
cycle_length = 256  # Define the length of the cycles


def extract_ecg_cycles(recording, frequency, target_length=cycle_length, cycle_num=5, overlap=3):
    """
    Redefined extract method to extract all leads in ECG record
    Returns a 12 row matrix, one row for each lead
    Each row is a list of segmented cycles
    """

    cycles_matrix = []

    # Adjust step size for overlapping windows
    step_size = cycle_num - overlap

    for lead in recording:
        if len(lead) != fixed_length:
            return None

        r_peak_indices, filtered_signal = denoise_find_r_peaks_elgendi(lead, frequency)

        cycles = []

        for i in range(cycle_num, len(r_peak_indices), step_size):
            start = r_peak_indices[i - cycle_num]
            end = r_peak_indices[i]
            cycle = lead[start:end]
            if len(cycle) > 0:
                resampled_cycle = resample_cycle(cycle, target_length * cycle_num)
                cycles.append(resampled_cycle)

        cycles_matrix.append(cycles)

    

    return cycles_matrix



def load_data(paths, disease_labels=None, data_type="normal", max_circle=None, cycle_num=1, overlap=0):
    """
    Loads 12 lead ECG data of passed dat type from passed list of paths
    Returns a list of 12 row matrices, where each row corresponds to a lead and contains a dict corresdponding to each cycle in the lead
    """

    cnt = 0
    data_matrices = []
    used_paths = []

    for path in paths:
        path_name_recorded = 0
        header_files, recording_files = find_all_challenge_files(path)
        length = len(header_files)

        for i in range(length):
            # Matrix containing 12 lead ECG data
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

            # Extract ECG file into a 12-row matrix of cycles
            all_lead_cycles = extract_ecg_cycles(mat['val'], frequency, cycle_length, cycle_num, overlap)

            if all_lead_cycles is None:
                continue
                
            diagnosis = get_labels(header)
            if data_type == "normal":
                is_normal = 1 if len(diagnosis) == 1 and '426783006' in diagnosis else 0
                if is_normal == 1:
                    cnt += 1
                    if path_name_recorded == 0:
                        used_paths.append(path)
                        path_name_recorded = 1

                    for lead in all_lead_cycles:
                        rows = []
                        for cycle in lead:
                            row_data = {'diagnosis': diagnosis}
                            for j in range(len(cycle)):
                                row_data[j + 1] = cycle[j]
                            rows.append(row_data)
                        data_matrix.append(rows)

                    # Add matrix to list of matrices
                    data_matrices.append(data_matrix)
                        

            elif data_type == "abnormal":
                is_normal = 0 if '426783006' not in diagnosis else 1
                if is_normal == 0:
                    cnt += 1
                    if path_name_recorded == 0:
                        used_paths.append(path)
                        path_name_recorded = 1

                    for lead in all_lead_cycles:
                        rows = []
                        for cycle in lead:
                            row_data = {'diagnosis': diagnosis}
                            for j in range(len(cycle)):
                                row_data[j + 1] = cycle[j]
                            rows.append(row_data)
                        data_matrix.append(rows)

                    # Add matrix to list of matrices
                    data_matrices.append(data_matrix)

            elif data_type == "single_abnormal":
                is_normal = 0 if len(diagnosis) == 1 and '426783006' not in diagnosis else 1
                if is_normal == 0 and int(diagnosis[0]) in disease_labels:
                    cnt += 1
                    if path_name_recorded == 0:
                        used_paths.append(path)
                        path_name_recorded = 1

                    for lead in all_lead_cycles:
                        rows = []
                        for cycle in lead:
                            row_data = {'diagnosis': diagnosis}
                            for j in range(len(cycle)):
                                row_data[j + 1] = cycle[j]
                            rows.append(row_data)
                        data_matrix.append(rows)

                    # Add matrix to list of matrices
                    data_matrices.append(data_matrix)

            if max_circle is not None and len(data_matrix[0]) >= max_circle:
                break

            

    print(f"Number of {data_type} records: {cnt}")
    return data_matrices, used_paths


def prep_data(matrices):
    """
    Takes list of matrices and prepares them as input for transformer model
    Separates labels from matrices, maintaining matching indices
    Returns:
        list of tensors
        corresponding labels
    """

    labels = []
    tensors = []

    for matrix in matrices:
        # Store diagnosis since it's the same for the whole matrix
        diagnosis = matrix[0][0]['diagnosis']
        labels.append(diagnosis)

        # Find max number of cycles in any particular lead
        max_lead_len = max([len(lead) for lead in matrix])

        # Pad leads with extra cycles of 0 values
        padded_leads = []
        for lead in matrix:
            lead_array = []
            for cycle in lead:
                cycle_array = []
                for i in range(1, 257):
                    cycle_array.append(cycle[i])
                lead_array.append(np.array(cycle_array))

            # Add padding
            diff = max_lead_len - len(lead)
            for i in range(diff):
                padding_cycle = np.zeros(256)
                lead_array.append(padding_cycle)

            padded_leads.append(np.array(lead_array))


        # Stack the padded leads to create a padded matrix
        padded_matrix = np.stack(padded_leads)

        # Create tensor and add to array
        tensor = torch.tensor(padded_matrix, dtype=torch.float32)
        tensors.append(tensor)



    return tensors, labels




if __name__ == "__main__":
    from const import *

    dataset_paths = [r"/Volumes/T7/College/GuoResearch/Transformer/data/physionet.org/files/challenge-2021/1.0.3/training/chapman_shaoxing"]
    data_test_1, dataset_test_1 = load_data(dataset_paths, data_type="normal", max_circle=10)
    data_test_2, dataset_test_2 = load_data(dataset_paths, diagnosis_chosen_0, data_type="abnormal", max_circle=10)
    data_test_3, dataset_test_3 = load_data(dataset_paths, diagnosis_chosen_0, data_type="single_abnormal", max_circle=10)
    print(data_test_1.head())
    print(dataset_test_1)
    print(data_test_2.head())
    print(dataset_test_2)
    print(data_test_3.head())
    print(dataset_test_3)
