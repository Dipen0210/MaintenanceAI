"""
Data Loader Utility for Predictive Maintenance System
Provides unified interfaces for loading MIMII, CWRU, and C-MAPSS datasets.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import librosa
from pathlib import Path
from typing import Tuple, List, Dict, Optional


class MIMIILoader:
    """Loader for MIMII DUE (Machine Investigation and Inspection) Dataset."""
    
    def __init__(self, base_path: str, machine_type: str = "fan"):
        """
        Args:
            base_path: Path to Data directory
            machine_type: One of 'fan', 'pump', 'valve', 'gearbox'
        """
        self.base_path = Path(base_path) / machine_type
        self.machine_type = machine_type
        
    def get_train_files(self) -> List[str]:
        """Get list of training files (normal sounds only)."""
        train_path = self.base_path / "train"
        return sorted(glob.glob(str(train_path / "*.wav")))
    
    def get_test_files(self, domain: str = "source") -> Tuple[List[str], List[int]]:
        """
        Get test files with labels.
        
        Args:
            domain: 'source' or 'target' for domain shift testing
            
        Returns:
            Tuple of (file_paths, labels) where 0=normal, 1=anomaly
        """
        test_path = self.base_path / f"{domain}_test"
        files = sorted(glob.glob(str(test_path / "*.wav")))
        
        labels = []
        for f in files:
            # MIMII file naming: *_normal_* or *_anomaly_*
            if "anomaly" in os.path.basename(f):
                labels.append(1)
            else:
                labels.append(0)
                
        return files, labels
    
    def load_audio(self, file_path: str, sr: int = 16000) -> np.ndarray:
        """Load a single audio file."""
        y, _ = librosa.load(file_path, sr=sr)
        return y


class CWRULoader:
    """Loader for CWRU Bearing Dataset."""
    
    LABEL_MAP = {
        'Normal': 0,
        'Ball_007': 1, 'Ball_014': 2, 'Ball_021': 3,
        'IR_007': 4, 'IR_014': 5, 'IR_021': 6,
        'OR_007': 7, 'OR_014': 8, 'OR_021': 9
    }
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Path to Data/CWRU directory
        """
        self.base_path = Path(base_path)
        
    def load_processed_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-processed feature CSV."""
        csv_path = self.base_path / "feature_time_48k_2048_load_1.csv"
        df = pd.read_csv(csv_path)
        
        # Extract features and labels
        feature_cols = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
        X = df[feature_cols].values
        
        # Map fault labels to integers
        y = df['fault'].apply(self._map_label).values
        
        return X.astype(np.float32), y.astype(np.int64)
    
    def load_cnn_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-processed CNN data from .npz file."""
        npz_path = self.base_path / "CWRU_48k_load_1_CNN_data.npz"
        data = np.load(npz_path)
        return data['X'], data['y']
    
    def _map_label(self, label: str) -> int:
        """Map string label to integer."""
        for key, val in self.LABEL_MAP.items():
            if key.lower() in label.lower():
                return val
        return 0  # Default to Normal
    
    def load_raw_mat(self, mat_file: str) -> np.ndarray:
        """Load raw .mat file and extract drive-end accelerometer signal."""
        mat = scipy.io.loadmat(mat_file)
        # CWRU .mat files have keys like 'X097_DE_time' for drive-end data
        for key in mat.keys():
            if 'DE_time' in key:
                return mat[key].flatten()
        return None


class CMAPSSLoader:
    """Loader for NASA C-MAPSS Turbofan Engine Degradation Dataset."""
    
    COLUMN_NAMES = ['unit', 'cycle', 'op1', 'op2', 'op3'] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: Path to Data/CMaps directory
        """
        self.base_path = Path(base_path)
        
    def load_train(self, subset: str = "FD001") -> pd.DataFrame:
        """Load training data for a subset."""
        file_path = self.base_path / f"train_{subset}.txt"
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.COLUMN_NAMES)
        return df
    
    def load_test(self, subset: str = "FD001") -> Tuple[pd.DataFrame, np.ndarray]:
        """Load test data and RUL labels."""
        test_path = self.base_path / f"test_{subset}.txt"
        rul_path = self.base_path / f"RUL_{subset}.txt"
        
        df = pd.read_csv(test_path, sep=r'\s+', header=None, names=self.COLUMN_NAMES)
        rul = pd.read_csv(rul_path, header=None).values.flatten()
        
        return df, rul
    
    def compute_rul(self, df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
        """Compute RUL for training data (decreasing from max cycle)."""
        df = df.copy()
        
        # Get max cycle per unit (engine)
        max_cycles = df.groupby('unit')['cycle'].max()
        
        # Compute RUL = max_cycle - current_cycle
        df['RUL'] = df.apply(lambda row: max_cycles[row['unit']] - row['cycle'], axis=1)
        
        # Cap RUL at max_rul (piecewise linear degradation model)
        df['RUL'] = df['RUL'].clip(upper=max_rul)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 50, 
                         feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM training.
        
        Args:
            df: DataFrame with RUL column computed
            sequence_length: Number of timesteps per sequence
            feature_cols: Columns to use as features (default: sensors only)
            
        Returns:
            X: shape (N, sequence_length, n_features)
            y: shape (N,) - RUL at end of each sequence
        """
        if feature_cols is None:
            feature_cols = [f'sensor_{i}' for i in range(1, 22)]
        
        sequences = []
        targets = []
        
        for unit_id in df['unit'].unique():
            unit_data = df[df['unit'] == unit_id]
            
            if len(unit_data) < sequence_length:
                continue
                
            features = unit_data[feature_cols].values
            rul_values = unit_data['RUL'].values
            
            # Create sliding windows
            for i in range(len(unit_data) - sequence_length + 1):
                sequences.append(features[i:i + sequence_length])
                targets.append(rul_values[i + sequence_length - 1])
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


# Convenience function
def get_loaders(data_dir: str) -> Dict:
    """Get all data loaders."""
    return {
        'mimii_fan': MIMIILoader(data_dir, 'fan'),
        'mimii_pump': MIMIILoader(data_dir, 'pump'),
        'mimii_valve': MIMIILoader(data_dir, 'valve'),
        'cwru': CWRULoader(os.path.join(data_dir, 'CWRU')),
        'cmapss': CMAPSSLoader(os.path.join(data_dir, 'CMaps'))
    }
