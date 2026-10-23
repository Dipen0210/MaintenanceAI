"""Measured baseline on available CWRU and NASA data, preserving legacy identities.

Legacy split/model-selection provenance is incomplete. These are reproducibility
checks and descriptive metrics, not independent real-plant validation.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.inference.fault_diagnoser import FaultDiagnoser
from src.inference.rul_estimator import RULEstimator
from src.preprocessing.rul_processor import RULPreprocessor


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    output={'scope':'Legacy datasets only; original model selection is not independently validated', 'vibration':{}, 'rul':{}}
    path=Path('Data/CWRU/CWRU_48k_load_1_CNN_data.npz')
    z=np.load(path); labels=LabelEncoder().fit_transform(z['labels'])
    _,x,_,y=train_test_split(z['data'],labels,test_size=.2,random_state=42,stratify=labels)
    model=FaultDiagnoser('trained_models/vibration_classifier.pth')
    start=time.perf_counter();pred=[]
    for image in x:pred.append(model.class_labels.index(model.predict(image)['fault_type']))
    output['vibration']={'accuracy':float(np.mean(np.array(pred)==y)),'count':len(y),
                         'elapsed_seconds':time.perf_counter()-start,'dataset_sha256':digest(path),
                         'artifact_sha256':digest('trained_models/vibration_classifier.pth'),
                         'split':'Legacy random-state 42 stratified 20% example split; recordings may overlap training'}
    for kind in ['predictor','transformer']:
        for subset in ['FD001','FD002','FD003','FD004']:
            path=Path(f'trained_models/rul_{kind}_{subset}.pth'); model=RULEstimator(str(path))
            processor=RULPreprocessor(sequence_length=model.sequence_length)
            frame=processor.load_data(f'Data/CMaps/test_{subset}.txt')
            true=np.loadtxt(f'Data/CMaps/RUL_{subset}.txt')
            start=time.perf_counter();pred=[]
            for unit in sorted(frame['unit'].unique()):
                seq=frame[frame['unit']==unit][model.feature_cols].to_numpy()
                pred.append(model.predict(seq)['rul_cycles'])
            output['rul'][path.stem]={'rmse_cycles':float(np.sqrt(np.mean((np.array(pred)-true)**2))),
                'count':len(pred),'elapsed_seconds':time.perf_counter()-start,'artifact_sha256':digest(path),
                'dataset_sha256':digest(f'Data/CMaps/test_{subset}.txt'),
                'ground_truth_sha256':digest(f'Data/CMaps/RUL_{subset}.txt')}
            print(path.stem,output['rul'][path.stem]['rmse_cycles'],flush=True)
    Path('docs/evaluation/legacy_baseline.json').write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
