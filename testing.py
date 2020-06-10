
from pystoi.stoi import stoi
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from scipy.io import wavfile
from pesq import pesq
#from pypesq import pesq
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

max_length = 100 #max length in time domain
num_of_features = 513
hop_length = 512
n_fft = 1024
sampling_frequency =16000
#NOISY_SIGNALS_DIR = "D:/RNN/RNN/data_set/noisy_audio_test/"
#CLEAN_SIGNALS_DIR = 'D:/RNN/RNN/data_set/clean_audio_test_30/'
CLEAN_SIGNALS_DIR = 'D:/UB_MS/Thesis/Seprating speech signals/DataSet/set/test_clean/'
NOISY_SIGNALS_DIR = 'D:/UB_MS/Thesis/Seprating speech signals/DataSet/Results/LSTM/one_noise_256/'
test_PESQ = 0
test_STOI = 0


for i in range (1000):

    noisy_s, sr = librosa.load(NOISY_SIGNALS_DIR +'enhanced_' + str(i) + '.wav',sr = sampling_frequency)
    clean_s, sr = librosa.load(CLEAN_SIGNALS_DIR +'clean' + str(i) + '.wav',sr = sampling_frequency)

    lengthc = len(clean_s)
    lengthn = len(noisy_s)
    if(lengthn > lengthc):
        noisy_s = noisy_s[:lengthc]
    else:
        clean_s = clean_s[:lengthn]
    print(i)
    print("pesq: ", pesq(sr, clean_s,noisy_s, 'wb'))
    print("stoi: ", stoi(clean_s, noisy_s, sr, extended=False))
    test_PESQ += pesq(sr, clean_s, noisy_s, 'wb')
    test_STOI += stoi(clean_s, noisy_s, sr, extended=False)

print("PESQ", test_PESQ / 1000)
    
print("STOI", test_STOI / 1000)




