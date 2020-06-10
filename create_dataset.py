import numpy as np
import h5py
import librosa.display
import os

n_file = 2500
max_length = 300 #max length in time domain
num_of_features = 513
hop_length = 512
n_fft = 1024
sampling_frequency = 16000
TEST_DIR = 'C:/Users/Zainab Altaweel/source/repos/RNN/RNN/data_set/test/'
CLEAN_SIGNALS_DIR = 'D:/UB_MS/Thesis/Seprating speech signals/DataSet/clean_part4/'
NOISY_SIGNALS_DIR = 'D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/noisy_part4/'

def creat_spectrograms(data_dir, ext):
    stft_abs_data = np.zeros((0, max_length, hop_length +1))
    stft_abs_pad = np.zeros((1, max_length, hop_length +1))

    #for file in os.listdir(NOISY_SIGNALS_DIR):
    for i in range (n_file):
        #read audio signals 
        s, sr = librosa.load(data_dir + ext + str(i+7501) + ".wav", sr = sampling_frequency)
        #find the stft of audio signals
        stft = librosa.stft(s, n_fft=n_fft, hop_length=hop_length) 
        print(i)
        #print("sampling rate: ", sr)
        stft_len = stft.shape[1]
        print("sequence length: ", stft_len)
        #print("spec length:", stft_len)
        #print("stft: ", stft)
        # keep the magnitude component only
        stft_abs = np.abs(stft) 
        #Padding zeros to make length 300
        stft_abs = np.pad(stft_abs, ((0,0),(0, max_length-stft_len)), 'constant')
        stft_abs_pad[0] = stft_abs.T
        stft_abs_data = np.append(stft_abs_data, stft_abs_pad, axis=0)
        print(stft_abs_data.shape)
    return(stft_abs_data)

noisy_data_set = np.zeros((n_file, max_length, hop_length +1), dtype='uint8')
clean_data_set = np.zeros((n_file, max_length, hop_length +1), dtype='uint8')

noisy_data_set = creat_spectrograms(NOISY_SIGNALS_DIR, "noisy")
clean_data_set = creat_spectrograms(CLEAN_SIGNALS_DIR, "clean")
#norm_value= max(np.amax(noisy_data_set), np.amax(clean_data_set))

#noisy_data_set = noisy_data_set/norm_value
#clean_data_set = clean_data_set/norm_value

hf = h5py.File(NOISY_SIGNALS_DIR + 'one_noise_dataset4.h5', 'w')
hf.create_dataset('noisy_signals', data=noisy_data_set)
hf.create_dataset('clean_signals', data=clean_data_set)
hf.close()

#hf = h5py.File(NOISY_SIGNALS_DIR + 'data.h5', 'r')
#print(hf.keys())
#n1 = hf.get('noisy_signals')
#n1 = np.array(n1)

