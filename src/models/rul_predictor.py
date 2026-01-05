
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
