"""
RUL Preprocessing Pipeline for NASA C-MAPSS Dataset.
Creates sliding window sequences for LSTM-based RUL prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
from pathlib import Path


class RULPreprocessor:
    """Preprocessor for C-MAPSS turbofan engine degradation data."""
    
    COLUMN_NAMES = ['unit', 'cycle', 'op1', 'op2', 'op3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    
    # Sensors that are constant or near-constant (can be dropped)
    DROP_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                    'sensor_16', 'sensor_18', 'sensor_19']
    
    def __init__(
        self,
        sequence_length: int = 50,
        max_rul: int = 125,
        normalize: bool = True
    ):
        """
        Args:
            sequence_length: Number of timesteps per sequence
            max_rul: Cap RUL values (piecewise linear degradation)
            normalize: Whether to standardize features
        """
        self.sequence_length = sequence_length
        self.max_rul = max_rul
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load C-MAPSS text file into DataFrame."""
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            header=None,
            names=self.COLUMN_NAMES
        )
        return df
    
    def compute_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute RUL labels for training data."""
        df = df.copy()
        
        # Get max cycle per engine unit
        max_cycles = df.groupby('unit')['cycle'].max()
        
        # Compute RUL = max_cycle - current_cycle
        df['RUL'] = df.apply(
            lambda row: max_cycles[row['unit']] - row['cycle'],
            axis=1
        )
        
        # Cap RUL at max_rul (piecewise linear)
        df['RUL'] = df['RUL'].clip(upper=self.max_rul)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns to use."""
        all_sensors = [f'sensor_{i}' for i in range(1, 22)]
        # Keep only informative sensors
        feature_cols = [s for s in all_sensors if s not in self.DROP_SENSORS]
        # Optionally include operational settings
        feature_cols = ['op1', 'op2', 'op3'] + feature_cols
        return feature_cols
    
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        if self.scaler is not None:
            feature_cols = self.get_feature_columns()
            self.scaler.fit(df[feature_cols])
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to features."""
        if self.scaler is not None:
            df = df.copy()
            feature_cols = self.get_feature_columns()
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM.
        
        Returns:
            X: shape (N, sequence_length, n_features)
            y: shape (N,) - RUL at end of each sequence
        """
        feature_cols = self.get_feature_columns()
        
        sequences = []
        targets = []
        
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id].sort_values('cycle')
            
            if len(unit_data) < self.sequence_length:
                continue
            
            features = unit_data[feature_cols].values
            rul_values = unit_data['RUL'].values
            
            # Create sliding windows
            for i in range(len(unit_data) - self.sequence_length + 1):
                seq = features[i:i + self.sequence_length]
                target = rul_values[i + self.sequence_length - 1]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def process_train_data(
        self,
        file_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Complete pipeline for training data."""
        df = self.load_data(file_path)
        df = self.compute_rul(df)
        self.fit_scaler(df)
        df = self.transform(df)
        return self.create_sequences(df)
    
    def process_test_data(
        self,
        test_file: str,
        rul_file: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process test data with ground truth RUL."""
        df = self.load_data(test_file)
        
        # Load ground truth RUL
        rul_true = pd.read_csv(rul_file, header=None).values.flatten()
        
        # Apply same normalization
        df = self.transform(df)
        
        feature_cols = self.get_feature_columns()
        
        # For test, take last sequence_length cycles per unit
        sequences = []
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id].tail(self.sequence_length)
            
            if len(unit_data) < self.sequence_length:
                # Pad if necessary
                pad_length = self.sequence_length - len(unit_data)
                padding = np.zeros((pad_length, len(feature_cols)))
                seq = np.vstack([padding, unit_data[feature_cols].values])
            else:
                seq = unit_data[feature_cols].values
            
            sequences.append(seq)
        
        return np.array(sequences, dtype=np.float32), rul_true.astype(np.float32)


class CMAPSSDataset(Dataset):
    """PyTorch Dataset for C-MAPSS data."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Shape (N, sequence_length, n_features)
            targets: Shape (N,)
        """
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return seq, target


def create_rul_dataloader(
    sequences: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for RUL data."""
    dataset = CMAPSSDataset(sequences, targets)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
