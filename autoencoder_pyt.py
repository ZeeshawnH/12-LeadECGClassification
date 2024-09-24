import torch
import torch.nn as nn


class DenseAutoEncoder(nn.Module):
    def __init__(self, sequence_length=256):
        super(DenseAutoEncoder, self).__init__()
        multiplier = sequence_length // 256

        self.encoder = nn.Sequential(
            nn.Linear(256 * multiplier, 128 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128 * multiplier, 64 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64 * multiplier, 32 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(32 * multiplier, 16 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(16 * multiplier, 8 * multiplier),
            nn.LeakyReLU(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8 * multiplier, 16 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(16 * multiplier, 32 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(32 * multiplier, 64 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(64 * multiplier, 128 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(128 * multiplier, 256 * multiplier),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self


class ConvFcAutoEncoder(nn.Module):
    def __init__(self, sequence_length=256):
        super(ConvFcAutoEncoder, self).__init__()
        multiplier = sequence_length // 256

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(8 * 32 * multiplier, 8 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(8 * multiplier, 16 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(16 * multiplier, 32 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(32 * multiplier, 64 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(64 * multiplier, 128 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(128 * multiplier, 256 * multiplier),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),  # B, 16, 250
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # B, 16, 125
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1),  # B, 8, 125
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),  # B, 8, 62
            nn.Flatten(1),  # B, 8 * 62 = 496
            nn.Linear(8 * 62, 16),
            nn.LeakyReLU(0.1),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 8 * 62),
            nn.LeakyReLU(0.1),
            nn.Unflatten(1, (8, 62)),  # B, 8, 62
            nn.ConvTranspose1d(8, 8, kernel_size=2, stride=2, padding=0, output_padding=0),  # B, 8, 125
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=0, output_padding=1),  # B, 16, 250
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=1, padding=1),  # B, 1, 250
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, x.size(2))
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output, final_state):
        attn_weights = torch.bmm(lstm_output, final_state.unsqueeze(2)).squeeze(2)
        soft_attn_weights = self.softmax(attn_weights)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state


class AttentionLSTMAutoEncoder(nn.Module):
    def __init__(self, sequence_length, hidden_dim, num_layers):
        super(AttentionLSTMAutoEncoder, self).__init__()
        multiplier = sequence_length // 256

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN Block
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # encoder
        self.encoder_lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.attention = SelfAttention(hidden_dim)

        # decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, 32, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32 * 64 * multiplier, 128 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(128 * multiplier, 256 * multiplier),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN Block
        x = self.cnn(x)

        # encoder
        x = x.permute(0, 2, 1)  # adjust the dimension for LSTM
        # print(x.shape)

        lstm_out, (h_n, c_n) = self.encoder_lstm(x)

        final_state = h_n[-1]

        # print(final_state.shape)
        # print(h_n[-1].shape)

        attn_out = self.attention(lstm_out, final_state)

        # decoder
        attn_out_expanded = attn_out.unsqueeze(1).repeat(1, x.size(1), 1)
        lstm_out, _ = self.decoder_lstm(attn_out_expanded)

        # lstm_out, _ = self.decoder_lstm(lstm_out)

        # print(lstm_out.shape)

        reconstructions = self.fc(lstm_out)

        # reconstructions = reconstructions.view(-1, reconstructions.size(1))
        return reconstructions


class LSTMAutoEncoder_MultiHeadAtt(nn.Module):
    def __init__(self, sequence_length, hidden_dim, num_layers):
        super(AttentionLSTMAutoEncoder, self).__init__()
        multiplier = sequence_length // 256

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN Block
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Self Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)

        # Decoder LSTM and Fully Connected layers
        self.decoder_lstm = nn.LSTM(hidden_dim, 32, num_layers, batch_first=True)

        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32 * 64 * multiplier, 128 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(128 * multiplier, 256 * multiplier),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN Block
        x = self.cnn(x)

        x = x.permute(0, 2, 1)  # Permute to (batch_size, seq_len, embed_dim)
        x, (_, _) = self.encoder_lstm(x)
        lstm_out = self.attention(x, x, x)[0]

        # attn_output, _ = self.attention(x, x, x)
        # lstm_out, (h_n, c_n) = self.encoder_lstm(attn_output)

        lstm_out, _ = self.decoder_lstm(lstm_out)

        reconstructions = self.fc(lstm_out)

        return reconstructions


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        attn_output, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_output))
        feedforward_output = self.feedforward(src)
        src = self.norm2(src + self.dropout(feedforward_output))
        return src


class TransformerAutoEncoder(nn.Module):
    def __init__(self, sequence_length, hidden_dim, num_heads, num_layers, feedforward_dim, dropout=0.1):
        super(TransformerAutoEncoder, self).__init__()
        multiplier = sequence_length // 256

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN Block
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        # Transformer Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        # Fully Connected layers
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(hidden_dim * 64 * multiplier, 128 * multiplier),
            nn.LeakyReLU(0.1),
            nn.Linear(128 * multiplier, 256 * multiplier),
            nn.Sigmoid()
        )

    def forward(self, x):
        # CNN Block
        x = self.cnn(x)

        # Transformer requires the shape (seq_len, batch_size, hidden_dim)
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_len, channels)
        x = x.permute(1, 0, 2)  # Change to (seq_len, batch_size, hidden_dim)

        # Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Transformer Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        x = x.permute(1, 0, 2)  # Change back to (batch_size, seq_len, hidden_dim)

        # Fully Connected layers
        reconstructions = self.fc(x)

        return reconstructions


# test model dimensions
if __name__ == '__main__':
    input_dim = 1  # 输入特征维度
    hidden_dim = 256  # LSTM隐层维度
    num_layers = 2  # LSTM层数

    num_heads = 8  # default for transformer
    num_layers_transformer = 4  # default for transformer
    feedforward_dim = 2048  # default for transformer
    batch_size = 32
    sequence_length = 256 * 3  # 序列长度

    model = DenseAutoEncoder(sequence_length)
    # print(model)
    test_input = torch.randn(batch_size, input_dim, sequence_length)
    test_output = model(test_input)
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape)

    model = ConvFcAutoEncoder(sequence_length)
    # print(model)
    test_input = torch.randn(batch_size, input_dim, sequence_length)
    test_output = model(test_input)
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape)

    model = AttentionLSTMAutoEncoder(sequence_length, hidden_dim, num_layers)
    print(model)
    test_input = torch.randn(batch_size, input_dim, sequence_length)
    test_output = model(test_input)
    print("Input shape:", test_input.shape)
    print("Output shape:", test_output.shape)

    # model = TransformerAutoEncoder(sequence_length, hidden_dim, num_heads, num_layers_transformer, feedforward_dim)
    # # print(model)
    # test_input = torch.randn(batch_size, input_dim, sequence_length)
    # test_output = model(test_input)
    # print("Input shape:", test_input.shape)
    # print("Output shape:", test_output.shape)

