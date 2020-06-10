
from scipy.io import wavfile
import os
import numpy as np
import librosa
from random import seed
from random import randint

seed(1)

INPUT_CLEAN_DIR = "D:/UB_MS/Thesis/Seprating speech signals/DataSet/set/test_clean/"
INPUT_NOISE_DIR = 'D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/'
OUTPUT_DIR = 'D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/noisy_test/'
sampling_frequency = 16000


def addNoise(audio,noise, snrTarget):
    sigPower = getPower(audio)
    noisePower = getPower(noise) 
    factor = (sigPower / noisePower ) / (10**(snrTarget / 20.0))  # noise Coefficient for target SNR
    return ( audio + noise * np.sqrt(factor) )

def getPower(clip):
    clip2 = clip.copy()
    clip2 = np.array(clip2) / (2.0**15)  # normalizing
    clip2 = clip2 **2
    return np.sum(clip2) / (len(clip2) * 1.0)



for i in range (2500):
    data_clean, rate_clean = librosa.load(INPUT_CLEAN_DIR + "clean" + str(i)+".wav")
    print("Data:" , data_clean)
    #rate_clean, data_clean = wavfile.read(CLEAN_SIGNALS_DIR + clean)
    data_noise, rate_noise = librosa.load(INPUT_NOISE_DIR + "n1.wav")
    print("noise: ", data_noise)
    length = len(data_clean)
    data_noise = data_noise[:length]
    snr_level = randint(0, 20)
    average = addNoise(data_clean, data_noise, snr_level)
    print(type(average))
    print("average: ", average)
    filename = '%s%s.wav' % (OUTPUT_DIR,"noisy" + str(i))
    librosa.output.write_wav(filename, average, sr = rate_clean)
    i = i+1

