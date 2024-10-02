import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

# --- SELayer ---
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
        # Squeeze: Global Average Pooling
        b, c, _ = x.size()
        y = F.adaptive_avg_pool1d(x, 1).view(b, c)
        # Excitation: Fully Connected Layers
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        # Scale: Channel-wise multiplication
        return x * y.expand_as(x)

# --- Residual Block ---
class ResidualBlock(nn.Module):
    """
    Residual Block with Depthwise Separable Convolution and SE Layer.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.activation = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.se = SELayer(out_channels)
        self.downsample = downsample
        self.final_activation = nn.ELU(inplace=True)
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.final_activation(out)

        return out

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        """
        Positional Encoding module injects information about the relative or absolute position of the tokens in the sequence.
        
        Args:
            embed_dim (int): The embedding dimension.
            max_len (int): The maximum length of the incoming sequence.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
        
        Returns:
            torch.Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# --- Transformer Encoder Module ---
class TransformerEncoderModule(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, num_layers=2, dropout=0.1):
        """
        Transformer Encoder module composed of multiple TransformerEncoderLayers.
        
        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            num_layers (int): Number of TransformerEncoderLayers.
            dropout (float): Dropout probability.
        """
        super(TransformerEncoderModule, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
    
    def forward(self, x):
        """
        Passes the input through the Transformer Encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, embed_dim]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = self.transformer_encoder(x)
        return x

# --- Complete Transformer-Based CNN-FC Classifier ---
class TransformerCNNFCClassifier(nn.Module):
    def __init__(self, num_classes=3, target_length=32, num_heads=8, num_transformer_layers=2):
        """
        Transformer-Based CNN-FC Classifier for 1D sequential data.
        
        Args:
            num_classes (int): Number of target classes for classification.
            target_length (int): The desired sequence length after adaptive pooling.
            num_heads (int): Number of attention heads in Transformer.
            num_transformer_layers (int): Number of TransformerEncoder layers.
        """
        super(TransformerCNNFCClassifier, self).__init__()
        
        # Initial Convolutional Block
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),  # [batch, 64, L/2]
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),  # [batch, 64, L/4]
            nn.Dropout(p=0.3)  # Dropout for regularization
        )
        
        # Residual Layers
        self.layer1 = self._make_layer(64, 128, blocks=2, stride=2)   # [batch, 128, L/8]
        self.layer2 = self._make_layer(128, 256, blocks=2, stride=2)  # [batch, 256, L/16]
        self.layer3 = self._make_layer(256, 512, blocks=2, stride=2)  # [batch, 512, L/32]
        
        # Adaptive Average Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)  # [batch, 512, target_length]
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embed_dim=512, max_len=target_length)
        
        # Transformer Encoder
        self.transformer_encoder = TransformerEncoderModule(embed_dim=512, num_heads=num_heads, num_layers=num_transformer_layers)
        
        # Global Context Attention
        self.global_attention = nn.Sequential(
            nn.Linear(512 * target_length, 1024),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Sigmoid()
        )
        
        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),  # [batch, 512 * target_length]
            nn.Linear(512 * target_length, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)  # Output layer without activation (logits)
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
                nn.GroupNorm(num_groups=8, num_channels=out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass of the Transformer-Based CNN-FC Classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, sequence_length]
        
        Returns:
            torch.Tensor: Logits tensor of shape [batch_size, num_classes]
        """
        # Initial Convolutional Block
        x = self.initial_conv(x)                  # [batch, 64, L/4]
        
        # Residual Layers
        x = self.layer1(x)                        # [batch, 128, L/8]
        x = self.layer2(x)                        # [batch, 256, L/16]
        x = self.layer3(x)                        # [batch, 512, L/32]
        
        # Adaptive Average Pooling
        x = self.adaptive_pool(x)                 # [batch, 512, target_length]
        
        # Permute and Make Contiguous
        x = x.permute(0, 2, 1).contiguous()       # [batch, target_length, 512]
        
        # Add Positional Encoding
        x = self.pos_encoder(x)                   # [batch, target_length, 512]
        
        # Transformer Encoder
        x = self.transformer_encoder(x)           # [batch, target_length, 512]
        
        # Permute Back and Make Contiguous
        x = x.permute(0, 2, 1).contiguous()       # [batch, 512, target_length]
        
        # Apply Global Context Attention
        x_flat = x.reshape(x.size(0), -1)         # [batch, 512 * target_length]
        
        attention_weights = self.global_attention(x_flat).unsqueeze(2)  # [batch, 512, 1]
        
        x = x * attention_weights.expand_as(x)     # [batch, 512, target_length]
        
        # Classification Head
        x = self.classifier(x)                     # [batch, num_classes]
        
        return x

