import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    """
    Combined CNN-LSTM model for trading
    """

    def __init__(self, input_size: int, sequence_length: int, hidden_size: int = 128):
        super(CNN_LSTM, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv_dropout = nn.Dropout(0.2)

        # Calculate output size after convolution and pooling
        conv_output_size = sequence_length // 2  # After pooling
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )

        # Output layers
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # Reshape for CNN (batch_size, features, sequence_length)
        x = x.permute(0, 2, 1)

        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.conv_dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.conv_dropout(x)

        # Reshape back for LSTM (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)

        # LSTM layers
        lstm_out, _ = self.lstm(x)

        # Take last output
        out = lstm_out[:, -1, :]
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc(out)

        # Remove softmax - CrossEntropyLoss applies it internally
        return out