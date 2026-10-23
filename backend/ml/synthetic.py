"""Synthetic conveyor signal domain. Never use these models for real machinery."""
import numpy as np

RATE=4096
SIZE=4096
FEATURES=['rms','crest_factor','kurtosis','energy_30hz','energy_60hz','energy_120hz','energy_high_frequency']
LABELS=['normal','imbalance_like','bearing_impulse_like']


def signal(seed,label='normal'):
    rng=np.random.default_rng(seed);t=np.arange(SIZE)/RATE
    phase=rng.uniform(0,2*np.pi);speed=rng.uniform(29.5,30.5)
    x=rng.uniform(.7,1.3)*np.sin(2*np.pi*speed*t+phase)+rng.normal(0,.12,SIZE)
    if label=='imbalance_like':
        x+=rng.uniform(1.5,3)*np.sin(2*np.pi*speed*t+phase)
        x+=rng.uniform(.5,1)*np.sin(4*np.pi*speed*t)
    elif label=='bearing_impulse_like':
        x+=rng.uniform(.6,1.1)*np.sin(2*np.pi*120*t)
        for start in rng.integers(0,SIZE-40,30):x[start:start+40]+=rng.uniform(3,6)*np.exp(-np.arange(40)/7)*np.cos(2*np.pi*.2*np.arange(40))
    elif label!='normal':raise ValueError('Unknown synthetic state')
    return x.astype(float)


def features(values):
    x=np.asarray(values,dtype=float)
    if x.shape!=(SIZE,) or not np.isfinite(x).all():raise ValueError('Synthetic model requires 4096 finite samples at 4096 Hz')
    centered=x-x.mean();rms=float(np.sqrt(np.mean(centered**2)))
    if rms<1e-8:raise ValueError('Flat signal is not a valid synthetic operating window')
    power=np.abs(np.fft.rfft(centered))**2;freq=np.fft.rfftfreq(len(x),1/RATE);total=max(float(power.sum()),1e-12)
    band=lambda low,high:float(power[(freq>=low)&(freq<=high)].sum()/total)
    return np.array([rms,float(np.max(np.abs(centered))/rms),float(np.mean(centered**4)/rms**4),band(27,33),band(57,63),band(117,123),band(400,1800)])


def spectrum(values,rate):
    x=np.asarray(values,dtype=float).reshape(-1);x=x-x.mean()
    amplitudes=np.abs(np.fft.rfft(x))/len(x);freq=np.fft.rfftfreq(len(x),1/rate)
    indices=np.argsort(amplitudes[1:])[-5:][::-1]+1
    return [{'frequency_hz':round(float(freq[i]),3),'amplitude':round(float(amplitudes[i]),6)} for i in indices]


def waveform_summary(values):
    x=np.asarray(values,dtype=float).reshape(-1);x=x-x.mean();std=max(float(x.std()),1e-9)
    return np.array([float(np.sqrt(np.mean(x*x))),float(np.max(np.abs(x))/std),float(np.mean(x**4)/std**4)])
