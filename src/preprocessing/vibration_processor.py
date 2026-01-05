"""
Vibration Preprocessing Pipeline for CWRU Bearing Dataset.
Extracts FFT and statistical features from raw vibration signals.
"""

import numpy as np
import scipy.io
import scipy.signal
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class VibrationPreprocessor:
    """Extracts features from vibration signals."""
    
    def __init__(self, window_size: int = 2048, overlap: float = 0.5):
        """
        Args:
            window_size: Number of samples per window
            overlap: Overlap ratio between windows (0.0 to 1.0)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.hop_size = int(window_size * (1 - overlap))
        
    def segment_signal(self, signal: np.ndarray) -> np.ndarray:
        """Segment signal into overlapping windows."""
        n_samples = len(signal)
        n_windows = (n_samples - self.window_size) // self.hop_size + 1
        
        segments = []
        for i in range(n_windows):
            start = i * self.hop_size
            end = start + self.window_size
            segments.append(signal[start:end])
            
        return np.array(segments)
    
    def extract_time_features(self, segment: np.ndarray) -> Dict[str, float]:
        """Extract time-domain statistical features."""
        return {
            'max': np.max(segment),
            'min': np.min(segment),
            'mean': np.mean(segment),
            'std': np.std(segment),
            'rms': np.sqrt(np.mean(segment ** 2)),
            'skewness': scipy.stats.skew(segment) if hasattr(scipy, 'stats') else 0,
            'kurtosis': scipy.stats.kurtosis(segment) if hasattr(scipy, 'stats') else 0,
            'crest_factor': np.max(np.abs(segment)) / np.sqrt(np.mean(segment ** 2)),
            'peak_to_peak': np.max(segment) - np.min(segment)
        }
    
    def extract_fft_features(self, segment: np.ndarray, fs: int = 48000) -> np.ndarray:
        """Extract FFT magnitude spectrum."""
        fft = np.fft.rfft(segment)
        magnitude = np.abs(fft)
        return magnitude
    
    def process_segment(self, segment: np.ndarray) -> np.ndarray:
        """Process a single segment - normalize for CNN input."""
        # Normalize to zero mean, unit variance
        segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
        return segment


class CWRUDataset(Dataset):
    """PyTorch Dataset for CWRU vibration data."""
    
    LABEL_MAP = {
        'Normal': 0,
        'Ball_007': 1, 'Ball_014': 2, 'Ball_021': 3,
        'IR_007': 4, 'IR_014': 5, 'IR_021': 6,
        'OR_007': 7, 'OR_014': 8, 'OR_021': 9
    }
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        preprocessor: Optional[VibrationPreprocessor] = None
    ):
        """
        Args:
            data: Array of signal segments, shape (N, window_size)
            labels: Array of integer labels, shape (N,)
            preprocessor: VibrationPreprocessor instance
        """
        self.data = data
        self.labels = labels
        self.preprocessor = preprocessor or VibrationPreprocessor()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        segment = self.data[idx]
        label = self.labels[idx]
        
        # Normalize segment
        segment = self.preprocessor.process_segment(segment)
        
        # Convert to tensor with channel dimension: [1, window_size]
        segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
        
        return segment_tensor, label


def load_cwru_from_mat(mat_dir: str, window_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CWRU data from .mat files and create labeled segments.
    
    Args:
        mat_dir: Directory containing .mat files
        window_size: Samples per segment
        
    Returns:
        Tuple of (segments, labels)
    """
    mat_dir = Path(mat_dir)
    preprocessor = VibrationPreprocessor(window_size=window_size)
    
    all_segments = []
    all_labels = []
    
    # File patterns and their labels
    file_patterns = {
        'Normal': ['*Normal*.mat'],
        'Ball_007': ['*B007*.mat'],
        'Ball_014': ['*B014*.mat'],
        'Ball_021': ['*B021*.mat'],
        'IR_007': ['*IR007*.mat'],
        'IR_014': ['*IR014*.mat'],
        'IR_021': ['*IR021*.mat'],
        'OR_007': ['*OR007*.mat'],
        'OR_014': ['*OR014*.mat'],
        'OR_021': ['*OR021*.mat'],
    }
    
    label_map = CWRUDataset.LABEL_MAP
    
    for label_name, patterns in file_patterns.items():
        for pattern in patterns:
            files = list(mat_dir.glob(pattern))
            for mat_file in files:
                try:
                    mat = scipy.io.loadmat(str(mat_file))
                    # Find the drive-end accelerometer key
                    for key in mat.keys():
                        if 'DE_time' in key:
                            signal = mat[key].flatten()
                            segments = preprocessor.segment_signal(signal)
                            
                            all_segments.extend(segments)
                            all_labels.extend([label_map[label_name]] * len(segments))
                            break
                except Exception as e:
                    print(f"Error loading {mat_file}: {e}")
    
    return np.array(all_segments), np.array(all_labels)


def create_vibration_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for vibration data."""
    dataset = CWRUDataset(data, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
