"""
RUL Estimation Inference Module.
Loads trained LSTM and predicts Remaining Useful Life from sensor sequences.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.rul_predictor import RULPredictor


class RULEstimator:
    """Estimates Remaining Useful Life from sensor time series."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        sequence_length: int = 50,
        max_rul: int = 125
    ):
        """
        Args:
            model_path: Path to trained .pth file
            device: 'cuda' or 'cpu' (auto-detect if None)
            sequence_length: Required sequence length
            max_rul: Maximum RUL value (used for health score scaling)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.max_rul = max_rul
        self.scaler = None
        self.feature_cols = None
        
        # Load model with metadata
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            input_size = checkpoint.get('input_size', 17)
            hidden_size = checkpoint.get('hidden_size', 64)
            num_layers = checkpoint.get('num_layers', 2)
            self.sequence_length = checkpoint.get('sequence_length', 50)
            self.feature_cols = checkpoint.get('feature_cols', None)
            
            # Reconstruct scaler if available
            if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                self.scaler = StandardScaler()
                self.scaler.mean_ = np.array(checkpoint['scaler_mean'])
                self.scaler.scale_ = np.array(checkpoint['scaler_scale'])
            
            self.model = RULPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback: assume default architecture
            self.model = RULPredictor(input_size=17)
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Preprocess sensor sequence for inference.
        
        Args:
            sequence: Shape (seq_len, n_features) or (n_features,) for single timestep
        """
        # Handle single timestep
        if sequence.ndim == 1:
            sequence = sequence.reshape(1, -1)
        
        # Pad or trim to sequence_length
        if len(sequence) < self.sequence_length:
            pad_length = self.sequence_length - len(sequence)
            padding = np.zeros((pad_length, sequence.shape[1]))
            sequence = np.vstack([padding, sequence])
        elif len(sequence) > self.sequence_length:
            sequence = sequence[-self.sequence_length:]
        
        # Apply scaler if available
        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)
        
        # Convert to tensor: [1, seq_len, n_features]
        tensor = torch.tensor(sequence, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, sequence: np.ndarray) -> Dict:
        """
        Predict RUL from sensor sequence.
        
        Args:
            sequence: Sensor readings, shape (seq_len, n_features)
            
        Returns:
            Dict with rul_cycles, health_score, status, and recommendation
        """
        # Preprocess
        tensor = self.preprocess(sequence)
        
        # Get prediction
        with torch.no_grad():
            rul_pred = self.model(tensor)
        
        rul_cycles = max(0, rul_pred.item())  # Ensure non-negative
        
        # Calculate health score (0-1, where 1 is healthy)
        health_score = min(1.0, rul_cycles / self.max_rul)
        
        # Determine status and recommendation
        if rul_cycles > 100:
            status = 'HEALTHY'
            recommendation = 'Continue normal operation. Schedule routine maintenance.'
        elif rul_cycles > 50:
            status = 'WARNING'
            recommendation = 'Plan maintenance within the next maintenance window.'
        elif rul_cycles > 20:
            status = 'CRITICAL'
            recommendation = 'Schedule immediate maintenance. Reduce operational load if possible.'
        else:
            status = 'IMMINENT_FAILURE'
            recommendation = 'STOP operation immediately. Perform emergency maintenance.'
        
        return {
            'rul_cycles': round(rul_cycles, 1),
            'health_score': round(health_score, 3),
            'status': status,
            'recommendation': recommendation
        }
    
    def predict_batch(self, sequences: List[np.ndarray]) -> List[Dict]:
        """Predict on multiple sequences."""
        results = []
        for seq in sequences:
            results.append(self.predict(seq))
        return results
    
    def get_health_color(self, health_score: float) -> str:
        """Get color code for health score visualization."""
        if health_score >= 0.8:
            return 'green'
        elif health_score >= 0.5:
            return 'yellow'
        elif health_score >= 0.2:
            return 'orange'
        else:
            return 'red'


def load_rul_estimator(model_path: str) -> RULEstimator:
    """Convenience function to load an RUL estimator."""
    return RULEstimator(model_path=model_path)
