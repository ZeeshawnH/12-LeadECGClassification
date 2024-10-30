import pandas as pd
import scipy
from scipy.io import loadmat

from helper_code import find_all_challenge_files, get_frequency, get_labels, get_num_samples
from utils import load_header_with_fallback, butterworth_elgendi_rpeak, denoise_find_r_peaks_elgendi
from scipy.signal import resample

import numpy as np
import matplotlib.pyplot as plt
from const import *
from custom_data_loader_12_lead import *
from utils import *

import torch

import os
import time


cycle_num = 1

time = time.strftime("%Y%m%d%H%M%S", time.localtime())
save_path = r".\outputs" + "\\" + time + "_cycle_" + str(cycle_num)+"all"

# create the directory
try:
    os.makedirs(save_path)
except FileExistsError:
    pass

# choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get paths
root_directory = r"/home/zahasnai/physionet.org/files/challenge-2021/1.0.3/training"
dataset_paths = find_subfolders(root_directory)
print(dataset_paths)
print("")

overlap = 1
max_circle = None

# Load data
cycles, periods = load_data(dataset_paths, max_circle=None,
                        cycle_num=cycle_num, overlap=overlap)

print(len(cycles))
print(len(periods))