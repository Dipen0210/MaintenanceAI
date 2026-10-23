"""Execute one real FD001 training epoch from the corrected notebook.

This is a training-path integration test, not a new champion or accuracy claim.
Existing checkpoints are never overwritten.
"""
import ast,json,math,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def run():
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    notebook=json.loads(Path('notebooks/03_Colab_RUL_Prediction_Training.ipynb').read_text())
    source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    env=dict(globals(),device='cpu',DATA_PATH='Data/CMaps',SAVE_PATH='artifacts/rul_training_smoke',
             SEQUENCE_LENGTH=80,MAX_RUL=125,D_MODEL=64,N_HEADS=4,N_LAYERS=3,D_FF=128,DROPOUT=.1,
             EPOCHS=1,BATCH_SIZE=64,LEARNING_RATE=1e-4,
             COLUMN_NAMES=['unit','cycle','op1','op2','op3']+[f'sensor_{i}' for i in range(1,22)],
             DROP_SENSORS=['sensor_1','sensor_5','sensor_6','sensor_10','sensor_16','sensor_18','sensor_19'])
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<rul-notebook>','exec'),env)
    result=env['train_subset']('FD001')
    path=Path(env['SAVE_PATH'])/'rul_transformer_FD001.pth'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert not set(saved['training_engine_ids']) & set(saved['validation_engine_ids'])
    assert len(saved['training_engine_ids'])==80 and len(saved['validation_engine_ids'])==20
    from src.inference.rul_estimator import RULEstimator
    model=RULEstimator(str(path))
    assert np.isfinite(model.predict(np.tile(model.scaler.mean_,(50,1)))['rul_cycles'])
    report={'purpose':'One-epoch actual notebook integration check; not a deployment model',
            'train_engines':80,'validation_engines':20,'test_engines':100,
            'epochs':1,'seed':42,'test_rmse_cycles':float(result['rmse']),
            'split_disjoint':True,'runtime_reload_verified':True}
    Path('docs/evaluation/rul_training_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':run()
