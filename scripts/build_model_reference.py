"""Input-distribution reference from the legacy CWRU training partition only."""
import hashlib,json
import numpy as np
from sklearn.model_selection import train_test_split
from backend.core.database import ROOT
from backend.ml.synthetic import waveform_summary


def main():
    path=ROOT/'Data/CWRU/CWRU_48k_load_1_CNN_data.npz';z=np.load(path)
    train,_,_,_=train_test_split(z['data'],z['labels'],test_size=.2,random_state=42,stratify=z['labels'])
    x=np.array([waveform_summary(v) for v in train])
    report={'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'count':len(train),
            'split':'Legacy training partition; random_state=42, stratified 80%. Recording independence remains unverified.',
            'features':['rms','crest_factor','kurtosis'],'mean':x.mean(0).tolist(),'std':x.std(0).tolist()}
    (ROOT/'models/cwru-reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
