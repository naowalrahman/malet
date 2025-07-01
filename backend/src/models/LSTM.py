import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    LSTM-based neural network for trading signal prediction
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state with proper device handling
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output
        out = out[:, -1, :]

        # Apply dropout and fully connected layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)

        # Remove softmax - CrossEntropyLoss applies it internally
        return out