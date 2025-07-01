import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    """
    Transformer-based model for trading
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Multi-layer head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 2)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(d_model // 2)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Scale by sqrt(d_model) as per original transformer paper
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer encoder
        transformer_out = self.transformer(x)

        # Use last token instead of global average pooling for time series
        out = transformer_out[:, -1, :]  # Shape: (batch_size, d_model)
        
        # Apply layer norm
        out = self.layer_norm(out)
        
        # Multi-layer classification head
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # The original transformer formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it moves with the model to GPU/CPU
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)