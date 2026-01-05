"""
Vibration Fault Diagnosis Inference Module.
Loads trained classifier and diagnoses bearing faults from vibration signals.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict

# Import model architecture
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.vibration_classifier import VibrationClassifier


class FaultDiagnoser:
    """Diagnoses bearing faults from vibration signals."""
    
    # Default class labels (should match training)
    DEFAULT_LABELS = [
        'Normal',
        'Ball_007', 'Ball_014', 'Ball_021',
        'IR_007', 'IR_014', 'IR_021',
        'OR_007', 'OR_014', 'OR_021'
    ]
    
    # Human-readable fault descriptions
    FAULT_DESCRIPTIONS = {
        'Normal': 'No fault detected - bearing is healthy',
        'Ball_007': 'Ball fault (0.007" diameter)',
        'Ball_014': 'Ball fault (0.014" diameter)',
        'Ball_021': 'Ball fault (0.021" diameter)',
        'IR_007': 'Inner race fault (0.007" diameter)',
        'IR_014': 'Inner race fault (0.014" diameter)',
        'IR_021': 'Inner race fault (0.021" diameter)',
        'OR_007': 'Outer race fault (0.007" diameter)',
        'OR_014': 'Outer race fault (0.014" diameter)',
        'OR_021': 'Outer race fault (0.021" diameter)'
    }
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        class_labels: Optional[List[str]] = None,
        window_size: int = 2048
    ):
        """
        Args:
            model_path: Path to trained .pth file
            device: 'cuda' or 'cpu' (auto-detect if None)
            class_labels: List of class label names
            window_size: Expected input window size
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = class_labels or self.DEFAULT_LABELS
        self.window_size = window_size
        self.num_classes = len(self.class_labels)
        
        # Load model
        self.model = VibrationClassifier(num_classes=self.num_classes)
        
        # Try loading with metadata first
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'label_encoder_classes' in checkpoint:
                self.class_labels = checkpoint['label_encoder_classes']
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """Preprocess vibration signal for inference."""
        # Ensure correct length
        if len(signal) > self.window_size:
            signal = signal[:self.window_size]
        elif len(signal) < self.window_size:
            signal = np.pad(signal, (0, self.window_size - len(signal)))
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # Convert to tensor: [1, 1, window_size]
        tensor = torch.tensor(signal, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, signal: np.ndarray) -> Dict:
        """
        Predict fault type from vibration signal.
        
        Args:
            signal: 1D numpy array of vibration samples
            
        Returns:
            Dict with fault_type, confidence, description, and all_probs
        """
        # Preprocess
        tensor = self.preprocess(signal)
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        # Get top prediction
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        confidence = confidence.item()
        
        fault_type = self.class_labels[pred_idx]
        description = self.FAULT_DESCRIPTIONS.get(fault_type, 'Unknown fault')
        
        # Get all probabilities
        all_probs = {
            self.class_labels[i]: probs[0, i].item()
            for i in range(self.num_classes)
        }
        
        return {
            'fault_type': fault_type,
            'confidence': confidence,
            'description': description,
            'is_faulty': fault_type != 'Normal',
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, signals: List[np.ndarray]) -> List[Dict]:
        """Predict on multiple signals."""
        results = []
        for signal in signals:
            results.append(self.predict(signal))
        return results
    
    def get_severity(self, fault_type: str) -> str:
        """Get fault severity level."""
        if fault_type == 'Normal':
            return 'None'
        elif '007' in fault_type:
            return 'Minor'
        elif '014' in fault_type:
            return 'Moderate'
        elif '021' in fault_type:
            return 'Severe'
        return 'Unknown'


def load_fault_diagnoser(model_path: str) -> FaultDiagnoser:
    """Convenience function to load a fault diagnoser."""
    return FaultDiagnoser(model_path=model_path)
