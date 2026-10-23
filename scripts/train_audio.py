"""Reproducible MIMII Isolation Forest baseline; no labels from test sets tune it.

Run from repository root: python -m scripts.train_audio
Training and calibration split by filename hash inside the normal training set.
All source/target test observations are reserved for final metrics.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from src.inference.anomaly_detector import AudioFeatures, FEATURE_VERSION


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_role(path):
    return 'calibration' if int(hashlib.sha256(Path(path).name.encode()).hexdigest()[:8], 16) % 5 == 0 else 'train'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Data')
    parser.add_argument('--output', default='trained_models')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(exist_ok=True, parents=True)
    cache = Path('artifacts/audio_features'); cache.mkdir(parents=True, exist_ok=True)
    extractor = AudioFeatures()
    for machine in ['fan', 'pump', 'valve']:
        folder = Path(args.data) / machine
        paths = sorted(folder.rglob('*.wav'))
        records = []
        def extract(p):
            sha = digest(p)
            target = cache / f'{FEATURE_VERSION}-{sha}.npy'
            if target.exists():
                feature = np.load(target, allow_pickle=False)
            else:
                feature = extractor.extract_features(str(p)); np.save(target, feature)
            return dict(path=str(p), sha256=sha, features=feature)
        print(f'{machine}: extracting {len(paths)} real WAV files', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, record in enumerate(pool.map(extract, paths)):
                records.append(record)
                if (i+1) % 200 == 0: print(f'{machine}: {i+1}/{len(paths)}', flush=True)
        artifact = dict(format='mimii_iforest_v1', feature_version=FEATURE_VERSION,
                        machine_type=machine, sample_rate=16000, threshold=.95,
                        dataset='MIMII DUE local public dataset', sections={})
        report = dict(machine_type=machine, feature_version=FEATURE_VERSION, seed=42,
                      threshold_policy='95th normal calibration percentile; fixed before test', sections={})
        manifest = []
        for section in ['00', '01', '02']:
            rows = [r for r in records if f'section_{section}_' in Path(r['path']).name]
            train = [r for r in rows if Path(r['path']).parent.name == 'train' and split_role(r['path']) == 'train']
            calibration = [r for r in rows if Path(r['path']).parent.name == 'train' and split_role(r['path']) == 'calibration']
            if len(train) < 100 or len(calibration) < 20: raise ValueError('Insufficient real normal training data')
            if any('_normal_' not in r['path'] for r in train+calibration): raise ValueError('Training must be normal only')
            pipeline = Pipeline([('scaler',StandardScaler()),('pca',PCA(n_components=32, random_state=42)),
                                 ('detector',IsolationForest(n_estimators=200, random_state=42, n_jobs=1))])
            pipeline.fit(np.stack([r['features'] for r in train]))
            reference = np.sort(-pipeline.score_samples(np.stack([r['features'] for r in calibration])))
            artifact['sections'][section] = dict(pipeline=pipeline, calibration_scores=reference)
            metrics = dict(train_count=len(train), calibration_count=len(calibration), test={})
            # Detect duplicate recordings across fitting/calibration/evaluation partitions.
            train_hashes={r['sha256'] for r in train}; calibration_hashes={r['sha256'] for r in calibration}
            if train_hashes & calibration_hashes: raise ValueError('Duplicate recording across train/calibration')
            for domain in ['source_test','target_test']:
                test = [r for r in rows if Path(r['path']).parent.name == domain]
                if {r['sha256'] for r in test} & (train_hashes | calibration_hashes): raise ValueError('Test recording leakage')
                scores = -pipeline.score_samples(np.stack([r['features'] for r in test]))
                percentiles = np.searchsorted(reference,scores,side='right')/len(reference)
                labels = [int('_anomaly_' in r['path']) for r in test]
                metrics['test'][domain] = dict(count=len(test), auc=float(roc_auc_score(labels,scores)),
                    balanced_accuracy=float(balanced_accuracy_score(labels,percentiles>.95)))
            report['sections'][section]=metrics
            for r in rows:
                role=split_role(r['path']) if Path(r['path']).parent.name=='train' else Path(r['path']).parent.name
                manifest.append(dict(path=r['path'],sha256=r['sha256'],role=role,section=section))
        artifact['dataset_manifest_sha256'] = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        target=output/f'audio_iforest_v1_{machine}.joblib'; temporary=target.with_suffix('.tmp')
        joblib.dump(artifact,temporary); temporary.replace(target)
        report['artifact_sha256']=digest(target)
        report['dataset_manifest_sha256']=artifact['dataset_manifest_sha256']
        report['records']=manifest
        Path(f'docs/evaluation/audio_{machine}.json').write_text(json.dumps(report,indent=2))
        print(machine, 'trained and evaluated:',report['sections'],flush=True)

if __name__ == '__main__': main()
