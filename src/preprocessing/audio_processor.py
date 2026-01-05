"""
Audio Preprocessing Pipeline for MIMII DUE Dataset.
Converts raw WAV files to Mel-spectrograms for anomaly detection.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
from pathlib import Path


class AudioPreprocessor:
    """Converts audio signals to Mel-spectrograms."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        target_frames: int = 64
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        
    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file and resample if necessary."""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y
    
    def to_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Convert audio signal to normalized Mel-spectrogram."""
        # Compute Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [0, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        # Resize to target frames
        mel_norm = self._resize_frames(mel_norm)
        
        return mel_norm
    
    def _resize_frames(self, mel: np.ndarray) -> np.ndarray:
        """Pad or crop to target frame size."""
        if mel.shape[1] < self.target_frames:
            # Pad with zeros
            pad_width = self.target_frames - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Center crop
            start = (mel.shape[1] - self.target_frames) // 2
            mel = mel[:, start:start + self.target_frames]
        return mel
    
    def process_file(self, file_path: str) -> np.ndarray:
        """Complete pipeline: load audio and convert to Mel-spectrogram."""
        y = self.load_audio(file_path)
        mel = self.to_mel_spectrogram(y)
        return mel


class MIMIIDataset(Dataset):
    """PyTorch Dataset for MIMII audio data."""
    
    def __init__(
        self,
        file_paths: List[str],
        labels: Optional[List[int]] = None,
        preprocessor: Optional[AudioPreprocessor] = None
    ):
        """
        Args:
            file_paths: List of WAV file paths
            labels: Optional labels (0=normal, 1=anomaly). If None, assumed all normal.
            preprocessor: AudioPreprocessor instance
        """
        self.file_paths = file_paths
        self.labels = labels if labels is not None else [0] * len(file_paths)
        self.preprocessor = preprocessor or AudioPreprocessor()
        
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Get Mel-spectrogram
        mel = self.preprocessor.process_file(file_path)
        
        # Convert to tensor with channel dimension: [1, n_mels, n_frames]
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        
        return mel_tensor, label


def create_audio_dataloader(
    file_paths: List[str],
    labels: Optional[List[int]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for audio data."""
    dataset = MIMIIDataset(file_paths, labels)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
