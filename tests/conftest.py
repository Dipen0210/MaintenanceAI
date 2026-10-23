"""Keep CPU inference bounded and numba caches within the project workspace."""
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(__file__).resolve().parents[1]/'artifacts/numba_cache'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import torch
torch.set_num_threads(2)
