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
dataset_paths = dataset_paths[3:4]
print(dataset_paths)
print("")

overlap = 1
max_circle = None

# Load data
cycles, periods = load_data(dataset_paths, max_circle=None,
                        cycle_num=cycle_num, overlap=overlap)

print(len(cycles))
print(len(periods))

import pandas as pd
from itertools import chain
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict


# rename_mapping = {
#     '164889003': '164890007'
# }

# Function to rename diagnosis codes within a list
# def rename_diagnosis_codes(diagnosis_list, mapping):
#     return [mapping.get(code, code) for code in diagnosis_list]

# Count number for each diagnosis tuple
counts = defaultdict(int)


# for cycle in cycles:
#     # Apply renaming function
#     diagnosis = rename_diagnosis_codes(cycle[0], rename_mapping)
#     # Convert to tuple for easier counting
#     diagnosis_tuple = tuple(diagnosis)
#     # Counts number of records with each diagnosis TUPLE, does NOT count for each diagnosis code
#     counts[diagnosis_tuple] += 1



# print(counts)

# # Identify diagnosis labels with at least 20 samples
# valid_diagnoses = [diagnosis for diagnosis, count in counts.items() if count >= 20]

# # Filter the DataFrame to include only valid diagnosis labels
# filtered_cycles = [cycle for cycle in cycles if tuple(rename_diagnosis_codes(cycle[0], rename_mapping)) in valid_diagnoses]

# Drop the auxiliary 'diagnosis_tuple' column
# data_matrices.drop(columns=['diagnosis_tuple'], inplace=True)

# Reset index if desired
# data_matrices.reset_index(drop=True, inplace=True)

# print(f"Number of records after filtering: {len(cycles)}")

# Verify the counts after filtering
# final_counts = Counter(tuple(matrix[0]) for matrix in filtered_matrices)
# print(final_counts)










# import torch
# from itertools import chain
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Ensure 'diagnosis' lists are non-empty
# # for cycle_data in cycles:
# #     for lead in cycle_data:
# #         if len(lead[0]) == 0:
# #             lead[0] = ['Normal']  # Assign 'Normal' if empty

# Get all unique diagnosis codes
all_codes = set(cycle_data[0][0] for cycle_data in cycles)

# Map diagnosis codes to integer labels
print(all_codes)
code_to_label = {code: idx for idx, code in enumerate(sorted(all_codes))}

# Create empty lists to hold the feature data (cycles and duration) and labels
X_cycles = []
X_duration = []
y = []

# Process each matrix in the list
for cycle in cycles:
    label = cycle[0][0]
    # labels = set([code_to_label[lead[0]] for lead in cycle])  # Assign the first diagnosis code's label
    
    lead_cycles = []
    lead_duration = []
    # Each lead contains cycle data as a dictionary
    for lead in cycle:
        
        # Extract cycle points ('point_1', 'point_2', ...) and cycle duration
        cycle_points = [lead[i] for i in range(len(lead) - 2)] # All points except 'cycle_duration'
        duration = lead[len(lead) - 1] # Duration

        # Store cycles and duration separately
        lead_cycles.append(cycle_points)
        lead_duration.append(duration)
    
    X_cycles.append(lead_cycles)
    X_duration.append(lead_duration)
    y.append(label)  # Corresponding label


# Convert lists to NumPy arrays for further processing
X_cycles = np.array(X_cycles)
X_duration = np.array(X_duration)
y = np.array(y)






# Convert lists to numpy
X = np.array(X_cycles)


print(f"X shape {X.shape}")
print(f"Y shape: {y.shape}")

# Split data into training and test sets (stratify to preserve label distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)

# Print mappings and data shapes for verification
print(f"Diagnosis to Label Mapping: {code_to_label}")
print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}, Shape of y_test: {y_test.shape}")


# print(X_train)

# Scale the cycle data and duration separately
scaler_cycles = MinMaxScaler()
scaler_duration = MinMaxScaler()

# Scale each sample's cycle and duration data individually
X_train_cycles_scaled = []
X_train_duration_scaled = []
X_test_cycles_scaled = []
X_test_duration_scaled = []

# Scale the training data
for sample_cycles, sample_duration in zip(X_train, X_duration):
    # Reshape cycle data for the scaler and then back to original shape
    scaled_cycles = scaler_cycles.fit_transform(np.array(sample_cycles))
    X_train_cycles_scaled.append(scaled_cycles)

    # Reshape duration data for the scaler (already 1D)
    scaled_duration = scaler_duration.fit_transform(np.array(sample_duration).reshape(-1, 1))
    X_train_duration_scaled.append(scaled_duration)

# Scale the test data
for sample_cycles, sample_duration in zip(X_test, X_duration):
    scaled_cycles = scaler_cycles.transform(np.array(sample_cycles))
    X_test_cycles_scaled.append(scaled_cycles)

    scaled_duration = scaler_duration.transform(np.array(sample_duration).reshape(-1, 1))
    X_test_duration_scaled.append(scaled_duration)

# Convert scaled cycles and durations into NumPy arrays
X_train_cycles_scaled = np.array(X_train_cycles_scaled)
X_train_duration_scaled = np.array(X_train_duration_scaled)

X_test_cycles_scaled = np.array(X_test_cycles_scaled)
X_test_duration_scaled = np.array(X_test_duration_scaled)

# Combine cycles and duration features for each sample without flattening
X_train_scaled = [np.hstack((cycles, duration)) for cycles, duration in zip(X_train_cycles_scaled, X_train_duration_scaled)]
X_test_scaled = [np.hstack((cycles, duration)) for cycles, duration in zip(X_test_cycles_scaled, X_test_duration_scaled)]

# Convert lists to arrays
X_train_scaled = np.array(X_train_scaled)
X_test_scaled = np.array(X_test_scaled)

# Further split training data into training and validation sets
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=23, stratify=y_train
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Optionally, expand dimensions if needed (e.g., for CNN input)
X_train_tensor = X_train_tensor.unsqueeze(1)  # Shape: (batch_size, 1, num_features)
X_val_tensor = X_val_tensor.unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(1)

# Create datasets
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
train_batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=train_batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=train_batch_size, shuffle=False
)

print("Done splitting datasets.")



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_classifier(model, train_loader, val_loader, epochs=100, lr=0.0001, patience=10, device='cuda'):
    """
    Train a classification model with early stopping based on validation loss.

    Args:
        model (nn.Module): The classification model to train.
        train_loader (DataLoader): DataLoader for training data, yielding (inputs, labels).
        val_loader (DataLoader): DataLoader for validation data, yielding (inputs, labels).
        epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        device (str): Device to train the model on ('cuda' or 'cpu').

    Returns:
        nn.Module: The trained model with the best validation performance.
        list: Training loss history.
        list: Validation loss history.
        list: Validation accuracy history.
    """
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels = batch  # Assuming each batch is a tuple (inputs, labels)
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(inputs)  # Outputs are logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_predictions.double() / total_predictions
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy.item())

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_val_accuracy:.4f}")

        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
            print("Validation loss decreased, saving the model...")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses, val_accuracies


def evaluate_classifier(model, test_loader, device='cuda'):
    """
    Evaluate a classification model on test data.

    Args:
        model (nn.Module): The trained classification model.
        test_loader (DataLoader): DataLoader for test data, yielding (inputs, labels).
        device (str): Device to perform evaluation on ('cuda' or 'cpu').

    Returns:
        float: Mean test loss.
        float: Test accuracy.
        list: List of reconstruction (classification) errors per sample.
        torch.Tensor: Model outputs for the last batch.
    """
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    test_loss = 0.0
    correct = 0
    total = 0
    classification_errors = []
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()

            outputs = model(inputs)  # [batch_size, num_classes]
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)

            # Collect classification errors (optional)
            # Here, we can define error as 1 - accuracy per sample, but CrossEntropyLoss is not per-sample
            # Alternatively, collect whether each sample was correctly classified
            errors = (preds != labels).float().cpu().numpy()
            classification_errors.extend(errors)

            # Optionally, collect all outputs and labels
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    mean_test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct.double() / total
    return mean_test_loss, test_accuracy.item(), classification_errors, torch.cat(all_outputs, dim=0)



# attention cnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A Residual Block as introduced in ResNet.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.se = SELayer(out_channels)  # Squeeze-and-Excitation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)  # Apply SE block

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.elu(out)

        return out


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) block.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Squeeze
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        # Excitation
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)


class GlobalAttention(nn.Module):
    """
    Global Attention mechanism that takes into account both ECG and duration features.
    """
    def __init__(self, ecg_feature_size, duration_feature_size, intermediate_size=128):
        super(GlobalAttention, self).__init__()
        self.fc1 = nn.Linear(ecg_feature_size + duration_feature_size, intermediate_size, bias=False)
        self.elu = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(intermediate_size, ecg_feature_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ecg_features, duration_features):
        """
        Args:
            ecg_features (torch.Tensor): Shape [batch_size, ecg_feature_size]
            duration_features (torch.Tensor): Shape [batch_size, duration_feature_size]
        x
        Returns:
            torch.Tensor: Attention weights of shape [batch_size, ecg_feature_size]
        """
        combined = torch.cat((ecg_features, duration_features), dim=-1)  # [batch_size, ecg + duration]
        print(f"Shape of combined in GlobalAttention: {combined.shape}")
        y = self.fc1(combined)  # [batch_size, intermediate_size]
        print(f"Shape of y1 in GlobalAttention: {y.shape}")
        y = self.elu(y)
        print(f"Shape of y2 in GlobalAttention: {y.shape}")
        y = self.dropout(y)
        print(f"Shape of y3 in GlobalAttention: {y.shape}")
        y = self.fc2(y)  # [batch_size, ecg_feature_size]
        print(f"Shape of y4 in GlobalAttention: {y.shape}")
        y = self.sigmoid(y)  # [batch_size, ecg_feature_size]
        print(f"Shape of y5 in GlobalAttention: {y.shape}")
        return y


class AttentionConvFcClassifier(nn.Module):
    def __init__(self, num_classes=3, target_length=32, cycle_num=2):
        """
        Enhanced ConvFcClassifier with Residual Blocks, SE layers, and Global Attention.

        Args:
            num_classes (int): Number of target classes for classification.
            target_length (int): The desired sequence length after adaptive pooling.
            cycle_num (int): Number of ECG cycles per sample.
        """
        super(AttentionConvFcClassifier, self).__init__()
        
        self.cycle_num = cycle_num
        self.cycle_length = 256  # Assuming fixed length of each cycle

        # Initial Convolution for ECG Data
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels= 12 * 8*cycle_num, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(8*cycle_num),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual Layers for ECG Data
        self.layer1 = self._make_layer(12 * 8*cycle_num, 12 * 16*cycle_num, blocks=2, stride=2)
        self.layer2 = self._make_layer(12 * 16*cycle_num, 12 * 32*cycle_num, blocks=2, stride=2)
        self.layer3 = self._make_layer(12 * 32*cycle_num, 12 * 64*cycle_num, blocks=2, stride=2)
        
        # Adaptive Pooling for ECG Data
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)
        
        # Global Context Attention for ECG Data and Duration
        # The attention now takes into account both ECG and duration features
        # ECG features size after adaptive pooling: 64*cycle_num * target_length
        # Duration features size after projection: 64*cycle_num
        self.global_attention = GlobalAttention(
            ecg_feature_size= 12 * 64*cycle_num * target_length,
            duration_feature_size= 12 * 64*cycle_num,
            intermediate_size= 12 * 128*cycle_num
        )
        
        # Projection Layer for Duration Feature
        self.duration_projection = nn.Sequential(
            nn.Linear(12 * cycle_num, 12 * 64 * cycle_num),
            nn.BatchNorm1d(12 * 64 * cycle_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear( 12 * 64*cycle_num * target_length +  12 * 64*cycle_num,  12 * 64*cycle_num),
            nn.BatchNorm1d( 12 * 64*cycle_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear( 12 * 64*cycle_num,  12 * 32*cycle_num),
            nn.BatchNorm1d( 12 * 32*cycle_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear( 12 * 32*cycle_num, num_classes)  # Output layer without activation (logits)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Creates a layer consisting of Residual Blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of Residual Blocks.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: A sequential container of Residual Blocks.
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        """
        Forward pass of the ConvFcClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes]
        """
        # Split the input into ECG cycles and duration
        # Assuming the last dimension is the sequence length, and duration is a separate feature
        # Modify this part based on the actual input format
        # Here, we assume that the duration is the last value in the sequence
        print(x.shape)
        ecg_data = x[:, :, :, :-self.cycle_num]  # All cycles data
        ecg_data = x.squeeze(1)
        duration = x[:, :, :, -self.cycle_num:]    # Duration feature, shape: [batch_size, 1]
        duration = duration.squeeze(1)
        print(duration.shape)

        # Reshape ECG data to have shape (batch_size, 1, cycles_length * cycle_num)
        # ecg_data = ecg_data.view(ecg_data.size(0), 1, -1)


        # Convolutional layers for ECG data
        ecg_features = self.initial_conv(ecg_data)
        print(f"Initial convolution shape: {ecg_features.shape}")

        ecg_features = self.layer1(ecg_features)
        print(f"Layer 1 convolution shape: {ecg_features.shape}")
        ecg_features = self.layer2(ecg_features)
        print(f"Layer 2 convolution shape: {ecg_features.shape}")
        ecg_features = self.layer3(ecg_features)
        print(f"Layer 3 convolution shape: {ecg_features.shape}")
        ecg_features = self.adaptive_pool(ecg_features)  # Shape: [batch_size, 64*cycle_num, target_length]
        print(f"After adaptive pool convolution shape: {ecg_features.shape}")

        # Flatten the ECG feature map
        ecg_flat = ecg_features.view(ecg_features.size(0), -1)  # Shape: [batch_size, 64*cycle_num * target_length]

        # Project duration feature
        duration = duration.squeeze()
        print(f"Shape of duration_proj {duration.shape}")
        duration_proj = self.duration_projection(duration).squeeze() # Shape: [batch_size, 64*cycle_num]
        print(f"Shape after projection {duration.shape}")

        # Compute attention weights using both ECG and duration features
        print(f"Shape of ecg_flat: {ecg_flat.shape}")
        print(f"Shape of duration_proj: {duration_proj.shape}")
        attention_weights = self.global_attention(ecg_flat, duration_proj)  # Shape: [batch_size, 64*cycle_num * target_length]
        print(f"Shape of attention_weights: {attention_weights.shape}")


        attention_weights = attention_weights.view(ecg_features.size(0), 64*self.cycle_num, -1)  # Shape: [batch_size, 64*cycle_num, target_length]
        print(f"Shape of attention_weights: {attention_weights.shape}")

        print(f"Shape of ecg_features: {ecg_features.shape}")
        ecg_features = ecg_features * attention_weights  # Apply attention
        print(f"Shape of ecg_features: {ecg_features.shape}")

        # Flatten after attention
        print(f"Shape of ecg_flat: {ecg_flat.shape}")
        ecg_flat = ecg_features.view(ecg_features.size(0), -1)  # Shape: [batch_size, 64*cycle_num * target_length]
        print(f"Shape of ecg_flat: {ecg_flat.shape}")

        # Concatenate ECG features with projected duration
        combined_features = torch.cat((ecg_flat, duration_proj), dim=-1)  # Shape: [batch_size, 64*cycle_num * target_length + 64*cycle_num]
        print(f"Shape of combined_features: {combined_features.shape}")

        # Classification head
        out = self.classifier(combined_features)  # Shape: [batch_size, num_classes]
        print(f"Shape of out: {out.shape}")

        return out



# Define model parameters
num_classes = len(code_to_label)  # Number of target classes
target_length = cycle_num * 8
print(num_classes)


print("About to train")

# Initialize model with parameters
atten_conv_fc_classifier = AttentionConvFcClassifier(num_classes=num_classes, target_length=target_length, cycle_num=cycle_num).to(device)

atten_conv_fc_classifier, atten_train_losses_conv_fc, atten_val_losses_conv_fc, atten_val_accuracies_conv_fc = train_classifier(
    model=atten_conv_fc_classifier, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    epochs=500, 
    patience=50, 
    lr=0.00005, 
    device=device
)