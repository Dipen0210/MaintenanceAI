"""
Audio Anomaly Detection Inference Module.
Loads trained autoencoder and computes anomaly scores from audio files.
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Union, Tuple, Optional

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.audio_autoencoder import AudioAutoencoder


class AudioAnomalyDetector:
    """Detects anomalies in machine audio using reconstruction error."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        threshold: float = 0.1,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        target_frames: int = 64
    ):
        """
        Args:
            model_path: Path to trained .pth file
            device: 'cuda' or 'cpu' (auto-detect if None)
            threshold: Anomaly threshold (scores above = anomaly)
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
            n_fft: FFT size
            hop_length: Hop length for STFT
            target_frames: Target number of time frames
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_frames = target_frames
        
        # Load model
        self.model = AudioAutoencoder()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess(self, audio_path: str) -> torch.Tensor:
        """Convert audio file to mel-spectrogram tensor."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Compute mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [0, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        
        # Resize to target frames
        if mel_norm.shape[1] < self.target_frames:
            pad_width = self.target_frames - mel_norm.shape[1]
            mel_norm = np.pad(mel_norm, ((0, 0), (0, pad_width)))
        else:
            mel_norm = mel_norm[:, :self.target_frames]
        
        # Convert to tensor: [1, 1, n_mels, target_frames]
        tensor = torch.tensor(mel_norm, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def compute_anomaly_score(self, audio_path: str) -> float:
        """
        Compute anomaly score for an audio file.
        
        Returns:
            Anomaly score (higher = more anomalous)
        """
        # Preprocess
        mel_tensor = self.preprocess(audio_path)
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed = self.model(mel_tensor)
        
        # Compute MSE as anomaly score
        mse = torch.mean((mel_tensor - reconstructed) ** 2).item()
        
        return mse
    
    def predict(self, audio_path: str) -> Tuple[bool, float, str]:
        """
        Predict if audio is anomalous.
        
        Returns:
            Tuple of (is_anomaly, score, label)
        """
        score = self.compute_anomaly_score(audio_path)
        is_anomaly = score > self.threshold
        label = "ANOMALY" if is_anomaly else "NORMAL"
        
        return is_anomaly, score, label
    
    def predict_batch(self, audio_paths: list) -> list:
        """Predict on multiple audio files."""
        results = []
        for path in audio_paths:
            is_anomaly, score, label = self.predict(path)
            results.append({
                'path': path,
                'is_anomaly': is_anomaly,
                'score': score,
                'label': label
            })
        return results


def load_audio_detector(
    model_path: str,
    threshold: float = 0.1
) -> AudioAnomalyDetector:
    """Convenience function to load an audio anomaly detector."""
    return AudioAnomalyDetector(model_path=model_path, threshold=threshold)
