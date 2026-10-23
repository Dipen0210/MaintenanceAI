
import torch
import torch.nn as nn

class RULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(RULPredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output: RUL value
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Input_Size]
        lstm_out, _ = self.lstm(x)

        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]

        rul = self.regressor(last_out)
        return rul


import math

class PositionalEncoding(nn.Module):
    """Adds positional information to sequence."""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRUL(nn.Module):
    """Transformer Encoder for RUL Prediction with Attention."""

    def __init__(self, input_size, d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling (learns which timesteps matter most)
        self.attention_weights = nn.Linear(d_model, 1)

        # Output regression
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]

        # Project to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Attention pooling (weighted average)
        attn_scores = torch.softmax(self.attention_weights(x), dim=1)
        x = torch.sum(x * attn_scores, dim=1)  # [batch, d_model]

        # Predict RUL
        return self.regressor(x)
