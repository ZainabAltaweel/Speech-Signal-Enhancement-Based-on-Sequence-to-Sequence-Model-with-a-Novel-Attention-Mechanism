import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math
import pickle
import librosa
from keras.models import Model
from keras.layers import Input, GRU, Dense, Multiply, Bidirectional
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.models import load_model
import time
import sys
import tensorflow as tf
from encoder import*
from BahdanauAttention import*
from decoder import*
from tensorflow.python.client import device_lib
from decoder_no_attention import*
from keras.utils import plot_model

def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp,len_samples, iterations):
    # refrence: https://github.com/bkvogel/griffin_lim
    """Reconstruct an audio signal from a magnitude spectrogram.
    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.
    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    time_slices = magnitude_spectrogram.shape[0]
    print("time slices: ", time_slices)

    #len_samples = int(time_slices*hopsamp + fft_size)

    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    print("x_construct shape: ", x_reconstruct.shape)
    
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        #print("n=:" ,n)
        reconstruction_spectrogram = librosa.stft(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram.T)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        #print("proposal_spectrogram shape ", proposal_spectrogram.shape)
        prev_x = x_reconstruct
        #print("prev_x shape = ", prev_x.shape)

        x_reconstruct = librosa.istft(proposal_spectrogram.T, win_length = fft_size, hop_length = hopsamp)
        length = len(x_reconstruct)
        prev_x = prev_x[:length]
        #print("x_reconstruct shape after:", x_reconstruct.shape )
        #print("prev_x shape = ", prev_x.shape)

        diff = np.sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        #print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct



DIR = "D:/UB_MS/Thesis/Seprating speech signals/DataSet/test_noisy2/"
model = load_model('C:/Users/Zainab Altaweel/source/repos/seq2seq_gru/seq2seq_gru/audio_s2s.h5')
num_of_features = 513
hop_length = 512
n_fft = 1024
sampling_frequency =16000
for i in range (1000):
    s, sr = librosa.load(DIR +'noisy' + str(i) + '.wav', sr = sampling_frequency)
    #print("sr= ", sr)
    stft = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
    stft_len = stft.shape[1]
    stft_abs = np.abs(stft)
    #stft_abs = np.pad(stft_abs, ((0,0),(0, max_length - stft_len)), 'constant')
    stft_seq=np.zeros((1,stft_len,513))
    stft_seq[0]=stft_abs.T
    print("stft seq shape: ", stft_seq.shape)
    #print(stft_seq[0])
    decoded_audio = model.predict(stft_seq)
    #decoded_audio = evaluate(stft_seq)
    decoded_output = np.asarray(decoded_audio)

    print("decoded_output",decoded_output[0].shape)
    s_pred = librosa.istft(decoded_output[0].T, win_length = 1024, hop_length = 512)
    signal_len = s_pred.shape[0]

    x_reconstruct = reconstruct_signal_griffin_lim(decoded_output[0], n_fft, hop_length,signal_len, 300)

    librosa.output.write_wav('D:/UB_MS/Thesis/Seprating speech signals/DataSet/Results/Attention/256_from_data/enhanced' + str(i) + '.wav', x_reconstruct, sr= sampling_frequency)
