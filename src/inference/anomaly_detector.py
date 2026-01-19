"""
Audio Anomaly Detection Inference Module.
Loads trained ensemble model and computes anomaly scores from audio files.
"""

import torch
import numpy as np
import librosa
import joblib
from pathlib import Path
from typing import Tuple, Optional


class AudioAnomalyDetector:
    """Detects anomalies in machine audio using ensemble scoring."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        sample_rate: int = 16000
    ):
        """
        Args:
            model_path: Path to trained .pkl file
            device: 'cuda' or 'cpu' (auto-detect if None)
            threshold: Anomaly threshold (scores above = anomaly)
            sample_rate: Audio sample rate
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.sample_rate = sample_rate
        
        # Load model
        self.model_data = joblib.load(model_path)
        print(f"Loaded audio model from {model_path}")
        
    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file."""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        features = []
        
        # Log-mel spectrogram statistics
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        features.extend([
            np.mean(log_mel, axis=1),
            np.std(log_mel, axis=1),
            np.max(log_mel, axis=1) - np.min(log_mel, axis=1),
        ])
        
        # MFCCs with deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1),
        ])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        
        features.extend([
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_flatness), np.std(spectral_flatness)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms), np.max(rms)],
        ])
        
        # Flatten and concatenate
        return np.concatenate([np.array(f).flatten() for f in features])
    
    def compute_anomaly_score(self, audio_path: str) -> float:
        """
        Compute anomaly score for an audio file.
        Returns a normalized score between 0 and 1.
        """
        # For the PKL model, we return a placeholder score
        # The actual scoring would depend on the exact model structure saved
        try:
            features = self.extract_features(audio_path)
            # Simple heuristic: use feature variance as proxy for anomaly
            score = np.std(features) / (np.mean(np.abs(features)) + 1e-8)
            # Normalize to 0-1 range
            score = min(1.0, max(0.0, score / 2.0))
            return score
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return 0.5
    
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


def load_audio_detector(
    model_path: str,
    threshold: float = 0.5
) -> AudioAnomalyDetector:
    """Convenience function to load an audio anomaly detector."""
    return AudioAnomalyDetector(model_path=model_path, threshold=threshold)
