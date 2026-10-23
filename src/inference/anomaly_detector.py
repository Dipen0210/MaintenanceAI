"""Versioned, fitted MIMII audio anomaly inference. No heuristic fallback."""
import joblib
import librosa
import numpy as np
from pathlib import Path
from sklearn.utils.validation import check_is_fitted

FEATURE_VERSION = "whole_file_logmel_mfcc_v1"


class AudioFeatures:
    sample_rate = 16000

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract features from audio file."""
        # Load audio
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as exc:
            raise ValueError("Cannot decode audio") from exc
        if len(y) < self.sample_rate or len(y) > 60 * self.sample_rate or not np.isfinite(y).all():
            raise ValueError("Audio must contain 1–60 seconds of finite samples")
        
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
    

class AudioAnomalyDetector(AudioFeatures):
    def __init__(self, model_path: str, device=None, threshold=None, sample_rate=16000):
        self.model_data = joblib.load(model_path)
        if self.model_data.get('format') != 'mimii_iforest_v1':
            raise ValueError("Not a fitted audio artifact; legacy metrics files cannot perform inference")
        if self.model_data.get('feature_version') != FEATURE_VERSION or sample_rate != 16000:
            raise ValueError("Incompatible audio preprocessing")
        self.sample_rate = sample_rate
        self.threshold = float(self.model_data['threshold'] if threshold is None else threshold)
        if not 0 <= self.threshold <= 1:
            raise ValueError("Invalid threshold")
        for entry in self.model_data['sections'].values():
            check_is_fitted(entry['pipeline'].named_steps['detector'])
            if len(entry['calibration_scores']) < 20:
                raise ValueError("Missing held-out normal calibration scores")

    def compute_anomaly_score(self, audio_path: str, section: str = '00') -> float:
        if section not in self.model_data['sections']:
            raise ValueError("Unsupported machine section")
        entry = self.model_data['sections'][section]
        features = self.extract_features(audio_path)
        if not np.isfinite(features).all():
            raise ValueError("Non-finite audio features")
        raw_score = -entry['pipeline'].score_samples(features.reshape(1, -1))[0]
        reference = np.asarray(entry['calibration_scores'])
        # Empirical normal-score percentile, NOT a fault probability.
        return float(np.searchsorted(reference, raw_score, side='right') / len(reference))

    def predict(self, audio_path: str, section: str = '00'):
        score = self.compute_anomaly_score(audio_path, section)
        is_anomaly = score > self.threshold
        return is_anomaly, score, 'ANOMALY' if is_anomaly else 'NORMAL'


def load_audio_detector(model_path: str, threshold=None):
    return AudioAnomalyDetector(model_path=model_path, threshold=threshold)
