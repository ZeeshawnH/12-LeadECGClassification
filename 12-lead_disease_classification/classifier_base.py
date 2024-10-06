import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DenseResidualBlock(nn.Module):
    """
    Residual Block with two Linear Layers and optional downsampling.
    """
    def __init__(self, in_features, out_features):
        super(DenseResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(out_features, out_features)
        
        # Define downsample layer if dimensions change
        if in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LeakyReLU(0.1)
            )
        else:
            self.downsample = nn.Identity()  # No change needed

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Adjust identity dimensions if necessary
        identity = self.downsample(identity)
        
        # Residual connection
        out += identity
        out = self.activation(out)
        return out


class DenseClassifier(nn.Module):
    def __init__(self, sequence_length=256, num_classes=3):
        """
        Enhanced DenseClassifier with Residual Blocks and Global Attention.

        Args:
            sequence_length (int): Length of the input sequence.
            num_classes (int): Number of target classes for classification.
        """
        super(DenseClassifier, self).__init__()
        
        # Calculate multiplier using ceiling to handle non-multiples of 256
        self.multiplier = math.ceil(sequence_length / 256)
        self.expected_size = 256 * self.multiplier  # e.g., 256 * 3 = 768 for sequence_length=756

        self.encoder = nn.Sequential(
            nn.Linear(self.expected_size, 128 * self.multiplier),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Residual Block 1: 128*m -> 64*m
            DenseResidualBlock(128 * self.multiplier, 64 * self.multiplier),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            # Residual Block 2: 64*m -> 32*m
            DenseResidualBlock(64 * self.multiplier, 32 * self.multiplier),
            nn.LeakyReLU(0.1),
            
            # Residual Block 3: 32*m -> 16*m
            DenseResidualBlock(32 * self.multiplier, 16 * self.multiplier),
            nn.LeakyReLU(0.1),
            
            # Residual Block 4: 16*m -> 8*m
            DenseResidualBlock(16 * self.multiplier, 8 * self.multiplier),
            nn.LeakyReLU(0.1)
        )

        self.classifier = nn.Linear(8 * self.multiplier, num_classes)

    def forward(self, x):
        """
        Forward pass of the DenseClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes]
        """
        # Ensure input has the expected sequence length
        if x.size(2) < self.expected_size:
            padding = self.expected_size - x.size(2)
            x = F.pad(x, (0, padding), 'constant', 0)  # Pad on the right
        elif x.size(2) > self.expected_size:
            x = x[:, :, :self.expected_size]  # Truncate to the expected size

        # Flatten the input
        x = x.view(x.size(0), -1)  # [batch_size, 256 * multiplier]

        # Encode
        encoded = self.encoder(x)  # [batch_size, 8 * multiplier]

        # Classify
        outputs = self.classifier(encoded)  # [batch_size, num_classes]
        return outputs




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
    


class ConvFcClassifier(nn.Module):
    def __init__(self, num_classes=3, target_length=32, cycle_num=2):
        """
        Enhanced ConvFcClassifier with Residual Blocks, SE layers, and Global Attention.

        Args:
            num_classes (int): Number of target classes for classification.
            target_length (int): The desired sequence length after adaptive pooling.
        """
        super(ConvFcClassifier, self).__init__()
        
        # Initial Convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16*cycle_num, kernel_size=7, stride=2, padding=3, bias=False),  # [batch, 64, L/2]
            nn.BatchNorm1d(16*cycle_num),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # [batch, 64, L/4]
        )
        
        # Residual Layers
        self.layer1 = self._make_layer(16*cycle_num, 32*cycle_num, blocks=2, stride=2)   # [batch, 128, L/8]
        self.layer2 = self._make_layer(32*cycle_num, 64*cycle_num, blocks=2, stride=2)  # [batch, 256, L/16]
        self.layer3 = self._make_layer(64*cycle_num, 128*cycle_num, blocks=2, stride=2)  # [batch, 512, L/32]
        
        # Adaptive Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)  # [batch, 512, target_length]
        
        # Global Context Attention
        self.global_attention = nn.Sequential(
            nn.Linear(128*cycle_num * target_length, 256*cycle_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256*cycle_num, 128*cycle_num),
            nn.Sigmoid()
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [batch, 512 * target_length]
            nn.Linear(128*cycle_num * target_length, 128*cycle_num),
            nn.BatchNorm1d(128*cycle_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128*cycle_num, 64*cycle_num),
            nn.BatchNorm1d(64*cycle_num),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64*cycle_num, num_classes)  # Output layer without activation (logits)
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
        # print(f"Initial Input Shape: {x.shape}")  # [batch, 1, L]
        x = self.initial_conv(x)        # [batch, 64, L/4]
        # print(f"After initial_conv: {x.shape}")
        
        x = self.layer1(x)              # [batch, 128, L/8]
        # print(f"After layer1: {x.shape}")
        
        x = self.layer2(x)              # [batch, 256, L/16]
        # print(f"After layer2: {x.shape}")
        
        x = self.layer3(x)              # [batch, 512, L/32]
        # print(f"After layer3: {x.shape}")
        
        x = self.adaptive_pool(x)       # [batch, 512, target_length]
        # print(f"After adaptive_pool: {x.shape}")
        
        # Apply Global Context Attention
        x_flat = x.view(x.size(0), -1)  # [batch, 512 * target_length]
        # print(f"After flatten: {x_flat.shape}")
        
        attention_weights = self.global_attention(x_flat).unsqueeze(2)  # [batch, 512, 1]
        # print(f"Attention Weights Shape: {attention_weights.shape}")
        
        x = x * attention_weights.expand_as(x)  # [batch, 512, target_length]
        # print(f"After applying attention weights: {x.shape}")
        
        x = self.classifier(x)           # [batch, num_classes]
        # print(f"After classifier: {x.shape}")
        
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Args:
            hidden_dim (int): Dimension of the hidden state.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate on attention weights.
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Define linear layers for query, key, and value
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Output linear layer
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Linear projections
        Q = self.query_linear(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key_linear(x)
        V = self.value_linear(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.softmax(scores)  # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # Concatenate heads
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, hidden_dim)  # [batch, seq_len, hidden_dim]
        
        # Final linear layer
        out = self.out_linear(context)  # [batch, seq_len, hidden_dim]
        
        return out

class AttentionLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_classes, 
                 target_length=32, 
                 bidirectional=True, 
                 dropout=0.3, 
                 num_heads=4,
                 cycle_num=2):
        """
        Args:
            hidden_dim (int): Hidden dimension for the LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of target classes for classification.
            target_length (int): The desired sequence length after adaptive pooling.
            bidirectional (bool): If True, use bidirectional LSTM.
            dropout (float): Dropout rate.
            num_heads (int): Number of attention heads.
        """
        super(AttentionLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # CNN Block
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32*cycle_num, kernel_size=5, stride=1, padding=2),  # [batch, 32, L]
            nn.BatchNorm1d(32*cycle_num),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [batch, 32, L/2]

            nn.Conv1d(in_channels=32*cycle_num, out_channels=64*cycle_num, kernel_size=3, stride=1, padding=1),  # [batch, 64, L/2]
            nn.BatchNorm1d(64*cycle_num),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # [batch, 64, L/4]

            nn.Conv1d(in_channels=64*cycle_num, out_channels=128*cycle_num, kernel_size=3, stride=1, padding=1),  # [batch, 128, L/4]
            nn.BatchNorm1d(128*cycle_num),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)   # [batch, 128, L/8]
        )
        
        # Adaptive Pooling to ensure fixed output length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)  # [batch, 128, target_length]
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=128*cycle_num,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-Head Self-Attention
        self.multihead_attn = MultiHeadSelfAttention(hidden_dim * self.num_directions, num_heads=num_heads, dropout=dropout)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, 256*cycle_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256*cycle_num, 128*cycle_num),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128*cycle_num, num_classes)  # Output layer without activation (logits)
        )
        
    def forward(self, x):
        """
        Forward pass of the AttentionLSTMClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]

        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes]
        """
        # CNN Block
        x = self.cnn(x)               # [batch, 128, L/8]
        
        # Adaptive Pooling
        x = self.adaptive_pool(x)     # [batch, 128, target_length]
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)        # [batch, target_length, 128]
        
        # LSTM Encoder
        lstm_out, _ = self.encoder_lstm(x)  # [batch, target_length, hidden_dim * num_directions]
        
        # Multi-Head Self-Attention
        attn_out = self.multihead_attn(lstm_out)  # [batch, target_length, hidden_dim * num_directions]
        
        # Residual Connection and Layer Normalization
        attn_out = self.layer_norm(attn_out + lstm_out)  # [batch, target_length, hidden_dim * num_directions]
        
        # Pooling: Global Average Pooling over the sequence length
        context = attn_out.mean(dim=1)  # [batch, hidden_dim * num_directions]
        
        # Classification Head
        logits = self.classifier(context)  # [batch, num_classes]
        
        return logits  # Raw logits for each class

