"""
Vibration Fault Diagnosis Inference Module.
Loads trained classifier and diagnoses bearing faults from vibration signals.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# Import model architectures
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.vibration_classifier import VibrationClassifier1D, VibrationClassifier2D


class FaultDiagnoser:
    """Diagnoses bearing faults from vibration signals or spectrograms."""
    
    # Default class labels
    DEFAULT_LABELS = [
        'Normal',
        'Ball_007', 'Ball_014', 'Ball_021',
        'IR_007', 'IR_014', 'IR_021',
        'OR_007', 'OR_014', 'OR_021'
    ]
    
    # Fault descriptions
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
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = class_labels or self.DEFAULT_LABELS
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get metadata
        if isinstance(checkpoint, dict):
            self.num_classes = checkpoint.get('num_classes', 10)
            self.data_type = checkpoint.get('data_type', checkpoint.get('model_type', '1D'))
            if 'label_encoder_classes' in checkpoint:
                self.class_labels = checkpoint['label_encoder_classes']
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            self.num_classes = len(self.class_labels)
            self.data_type = '1D'
            state_dict = checkpoint
        
        # Create appropriate model
        if self.data_type == '2D':
            self.model = VibrationClassifier2D(num_classes=self.num_classes)
        else:
            self.model = VibrationClassifier1D(num_classes=self.num_classes)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {self.data_type} vibration model with {self.num_classes} classes")
    
    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """Preprocess input signal or image."""
        signal = signal.astype(np.float32)
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        if self.data_type == '2D':
            # Expect 2D input (H, W) or (C, H, W)
            if len(signal.shape) == 2:
                tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            elif len(signal.shape) == 3:
                tensor = torch.tensor(signal).unsqueeze(0)  # [1, C, H, W]
            else:
                raise ValueError(f"Unexpected 2D input shape: {signal.shape}")
        else:
            # Expect 1D input (signal_length,)
            if len(signal.shape) == 1:
                tensor = torch.tensor(signal).unsqueeze(0).unsqueeze(0)  # [1, 1, L]
            else:
                raise ValueError(f"Unexpected 1D input shape: {signal.shape}")
        
        return tensor.to(self.device)
    
    def predict(self, signal: np.ndarray) -> Dict:
        """Predict fault type from vibration signal or spectrogram."""
        tensor = self.preprocess(signal)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
        
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_idx = pred_idx.item()
        confidence = confidence.item()
        
        fault_type = self.class_labels[pred_idx] if pred_idx < len(self.class_labels) else f'Class_{pred_idx}'
        description = self.FAULT_DESCRIPTIONS.get(fault_type, f'Fault type: {fault_type}')
        
        return {
            'fault_type': fault_type,
            'confidence': confidence,
            'description': description,
            'is_faulty': 'normal' not in fault_type.lower(),
        }
    
    def get_severity(self, fault_type: str) -> str:
        """Get fault severity level."""
        if 'Normal' in fault_type or 'normal' in fault_type:
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
