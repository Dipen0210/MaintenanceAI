"""Reproduce demo-only models with disjoint run seeds and untouched final test runs."""
import hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from backend.ml.synthetic import signal,features,LABELS,FEATURES
ROOT=Path(__file__).resolve().parents[1]


def dataset(start,n):
    return np.array([features(signal(start+i,LABELS[i%3])) for i in range(n)]),np.array([LABELS[i%3] for i in range(n)])


def main():
    train,y=dataset(0,900);val,yv=dataset(10000,300);test,yt=dataset(20000,600)
    cal,_=dataset(30000,300);cal=cal[np.arange(300)%3==0]
    experiments=[];entries=[]
    for trees in [64,128]:
        classifier=RandomForestClassifier(n_estimators=trees,max_depth=8,random_state=42,n_jobs=1).fit(train,y)
        anomaly=IsolationForest(n_estimators=trees,random_state=42,n_jobs=1).fit(train[y=='normal'])
        calibration=-anomaly.score_samples(cal)
        report={'trees':trees,'validation_accuracy':float(accuracy_score(yv,classifier.predict(val))),
                'test_accuracy':float(accuracy_score(yt,classifier.predict(test))),
                'test_anomaly_auc':float(roc_auc_score(yt!='normal',-anomaly.score_samples(test))),
                'confusion_matrix':confusion_matrix(yt,classifier.predict(test),labels=LABELS).tolist()}
        bundle={'format':'synthetic_conveyor_v1','classifier':classifier,'anomaly':anomaly,'calibration':calibration,
                'feature_names':FEATURES,'reference_mean':train.mean(0),'reference_std':train.std(0)}
        path=ROOT/f'trained_models/synthetic_conveyor_{trees}.joblib';path.parent.mkdir(exist_ok=True)
        joblib.dump(bundle,path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({'id':f'conveyor-synthetic-{trees}-{digest[:12]}','family':'conveyor-synthetic','version':digest[:12],
                        'kind':'condition','filename':path.name,'sha256':digest,'dataset_id':'synthetic-conveyor-v1',
                        'preprocessing_version':'conveyor-spectrum-v1','equipment_types':['conveyor_motor'],
                        'operating_states':['running'],'sample_rate':4096,'input_kind':'vibration',
                        'mode':'simulation','metrics':report,'uncertainty':'Class scores are uncalibrated; no validated uncertainty interval.',
                        'limitations':'Synthetic training and evaluation only. Not validated for physical motors or conveyor assets.'})
        experiments.append(report)
    champion=max(range(len(experiments)),key=lambda i:experiments[i]['validation_accuracy'])
    for i,e in enumerate(entries):e['active']=i==champion
    report={'dataset':'synthetic-conveyor-v1','train_seeds':[0,899],'validation_seeds':[10000,10299],
            'test_seeds':[20000,20599],'normal_calibration_seeds':[30000,30299],
            'selection':'Validation accuracy, smaller model on ties. Test runs never used for selection.',
            'scope':'Only the explicitly synthetic simulator distribution; not industrial validation.',
            'champion_id':entries[champion]['id'],'experiments':experiments}
    (ROOT/'docs/evaluation/synthetic-models.json').write_text(json.dumps(report,indent=2)+'\n')
    (ROOT/'models/synthetic-registry.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
