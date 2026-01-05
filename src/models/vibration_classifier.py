
import torch
import torch.nn as nn

class VibrationClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(VibrationClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Conv Block 2
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Conv Block 3
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Conv Block 4
            nn.Conv1d(64, 64, kernel_size=8, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 30, 128), # Adjust input size based on final conv output
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, 1, 2048]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
