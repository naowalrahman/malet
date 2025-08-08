import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    """
    GRU with same surrounding structure as LSTM
    """
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 2):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layer - same structure as LSTM but using GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Same classification head as LSTM for fair comparison
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state with proper device handling (same as LSTM)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, 
                        device=x.device, dtype=x.dtype)

        # Ensure RNN parameters are in a contiguous chunk to avoid warnings and improve performance
        self.gru.flatten_parameters()

        # Forward propagate GRU
        gru_out, _ = self.gru(x, h0)

        # Take the last output (same as LSTM)
        out = gru_out[:, -1, :]
        
        # Apply the same classification head as LSTM
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out