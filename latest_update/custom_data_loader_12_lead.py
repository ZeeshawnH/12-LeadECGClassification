import pandas as pd
import scipy
from scipy.io import loadmat

from helper_code import find_all_challenge_files, get_frequency, get_labels, get_num_samples
from utils import load_header_with_fallback, butterworth_elgendi_rpeak, denoise_find_r_peaks_elgendi
from scipy.signal import resample

import numpy as np
import matplotlib.pyplot as plt
from const import *

# def extract_ecg_cycles(recording, frequency, num_samples, target_length=256, cycle_num=5, overlap=3):

#     cycles_matrix = []
#     durations_matrix = []

#     for lead, i in enumerate(recording):
#         if len(lead) != num_samples:
#             return None, None

#         r_peak_indices, filtered_signal = butterworth_elgendi_rpeak(lead, frequency)

#         if r_peak_indices is None:
#             return None, None
        
#         peak_num = cycle_num + 1
#         cycles = []
#         durations = []

#         # Calculate step size based on overlap
#         step = cycle_num - overlap
#         if step <= 0:
#             step = 1

#         for index in range(0, len(r_peak_indices) - cycle_num, step):
#             # Extract the segment from the current R-peak to the R-peak at the end of the specified cycle number
#             start_idx = r_peak_indices[index]
#             end_idx = r_peak_indices[index + cycle_num]
#             current_ecg = lead[start_idx:end_idx]

#             # Resample the current ECG segment to have a uniform length
#             current_ecg = resample(current_ecg, target_length * cycle_num)

#             # Compute durations for each cycle in the segment
#             cycle_durations = []
#             for k in range(cycle_num):
#                 duration = (r_peak_indices[index + k + 1] - r_peak_indices[index + k]) / frequency
#                 cycle_durations.append(duration)

#             cycles.append(current_ecg)
#             durations.append(cycle_durations)
            
#         cycles_matrix.append(cycles)
#         durations_matrix.append(durations)
    

#     return cycles_matrix, durations_matrix

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
    cycles = []
    duration_matrix = []

    for path in paths:
        header_files, recording_files = find_all_challenge_files(path)
        length = len(header_files)

        for i in range(length):

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

            all_lead_cycles, cycle_durations = extract_ecg_cycles(
                mat['val'], frequency, num_samples, target_length=256, cycle_num=cycle_num, overlap=overlap
            )
            if all_lead_cycles is None:
                continue

            diagnosis = get_labels(header)

            if len(diagnosis) == 1:
                diag_code = int(diagnosis[0])

                if diag_code in diagnosis_lead_1:
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

                ## TODO Do we want to pad the shorter leads here or later?
                max_lead_length = max(len(lead) for lead in all_lead_cycles)
                # Get length of a cycle for padding with 0s later
                if len(all_lead_cycles) > 0:
                    cycle_length = all_lead_cycles[0][0]
                else:
                    continue
                # Prepare rows for each cycle
                for cycle_idx in range(max_lead_length):
                    cycle_data = []
                    durations = []
                    for lead_idx in range(12):
                        if cycle_idx <  len(all_lead_cycles[lead_idx]):
                            cycle = all_lead_cycles[lead_idx][cycle_idx]
                        else:
                            cycle = [0.0] * (cycle_num * cycle_length)

                        if cycle_idx < len(cycle_durations[lead_idx]):
                            duration = cycle_durations[lead_idx][cycle_idx]
                        else:
                            duration = 0.0

                        row_data = [diag_label] + cycle + [duration]
                        durations.append(duration)
                        cycle_data.append(row_data)
                    duration_matrix.append(durations)
                    cycles.append(cycle_data)


                

            if max_circle is not None and len(cycles) >= max_circle:
                break

    print(f"Number of {data_type} records: {cnt}")
    return cycles, duration_matrix