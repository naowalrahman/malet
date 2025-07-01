import torch
import torch.nn as nn
import math


class Transformer(nn.Module):
    """
    Transformer-based model for trading
    """

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer
        transformer_out = self.transformer(x)

        # Global average pooling
        out = transformer_out.mean(dim=1)
        out = self.layer_norm(out)

        # Classification
        out = self.dropout(out)
        out = self.fc(out)

        # Remove softmax - CrossEntropyLoss applies it internally
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)