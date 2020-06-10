import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math
import pickle
import tensorflow as tf
import librosa
from keras.models import Model
from keras.layers import Input, GRU, Dense, Multiply, Bidirectional
from keras_self_attention import SeqSelfAttention
from keras.models import load_model
from keras.optimizers import Adam
import time
import keras_attention
import sys
from BahdanauAttention import*
from tensorflow.python.client import device_lib
from keras.utils import plot_model
import tensorflow.keras
from keras import backend as K
import keras_attention

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

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

TEST_DIR = 'C:/Users/Zainab Altaweel/source/repos/RNN/RNN/data_set/test/'
CLEAN_SIGNALS_DIR = 'C:/Users/Zainab Altaweel/source/repos/RNN/RNN/data_set/clean_audio_test/'
NOISY_SIGNALS_DIR = 'D:/RNN/RNN/data_set/noisy_audio_test/'
DIR = "D:/UB_MS/Thesis/Seprating speech signals/DataSet/set/test/"
BATCH_SIZE = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
units = 256  # Latent dimensionality of the encoding space.

n_file = 10000
max_length = 300 #max length in time domain
num_of_features = 513
hop_length = 512
n_fft = 1024
sampling_frequency = 16000


noisy_data_set = np.zeros((0, max_length, hop_length +1), dtype='uint8')
clean_data_set = np.zeros((0, max_length, hop_length +1), dtype='uint8')
test_data_set = np.zeros((0, max_length, hop_length +1), dtype='uint8')



for i in range(4):
    print(i)
    hf = h5py.File('D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/one_noise_dataset' + str(i+1) + '.h5', 'r')
    noisy_data = (np.array(hf.get('noisy_signals'))).astype(np.float32)
    clean_data = (np.array(hf.get('clean_signals'))).astype(np.float32)
    noisy_data_set = np.append(noisy_data_set, noisy_data, axis=0)
    clean_data_set = np.append(clean_data_set, clean_data, axis=0)



#hf = h5py.File( 'D:/dataset.h5', 'r')
#print(hf.keys())
#noisy_data_set = hf.get('noisy_signals')
#noisy_data_set = (np.array(noisy_data_set)).astype(np.float32)
#clean_data_set = hf.get('clean_signals')
#clean_data_set = (np.array(clean_data_set)).astype(np.float32)


encoder_input_data = noisy_data_set
decoder_target_data = clean_data_set

encoder_input_data = noisy_data_set
decoder_target_data = clean_data_set


print("noisy data shape: ", noisy_data_set.shape)
print("clean data shape: ", clean_data_set.shape)
BUFFER_SIZE = noisy_data_set.shape[0]
print("buffer size: ", BUFFER_SIZE)


#---------------------------------My Model----------------------------------#
#---------------------------------Encoder-----------------------------------#
encoder_inputs = tf.keras.layers.Input(shape=(None, num_of_features))
print("input shape: ", encoder_inputs.shape)

encoder_gru1 = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)
encoder_gru2 = tf.keras.layers.GRU(units, return_sequences=True, return_state=True)

#enc_output = encoder_dense(encoder_inputs)
enc_output, enc_hidden = encoder_gru1(encoder_inputs)


#---------------------------------Attention---------------------------------#
attention_layer = BahdanauAttention(10)
context_vector, attention_weights = attention_layer(enc_hidden, enc_output)


#print("shape of encoder output: ", enc_output.shape)

concat = tf.keras.layers.Concatenate()
enc_output = concat([context_vector, enc_output])

#print("shape of encoder output: ", enc_output.shape)




#----------------------------------Decoder-----------------------------------#
#decoder_inputs = tf.keras.Input(shape=(None, num_of_features))
decoder_gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
decoder_dense2 = tf.keras.layers.Dense(num_of_features, activation='tanh')
decoder_dense = tf.keras.layers.Dense(units, activation='sigmoid')
decoder_multiply = tf.keras.layers.Multiply()

decoder_outputs, decoder_state = decoder_gru(enc_output)
#decoder_outputs,_ = decoder_gru(enc_output)
decoder_outputs = decoder_dense2(decoder_outputs)
#decoder_outputs = decoder_dense2(decoder_outputs)
decoder_outputs = decoder_multiply([decoder_outputs, encoder_inputs])

#print ('Decoder output shape: (batch_size, vocab size) {}'.format(decoder_outputs.shape))


model = tf.keras.Model(encoder_inputs, decoder_outputs)
model.summary()
start_time = time.perf_counter()
opt= tf.keras.optimizers.Adam()
model.compile(loss='mean_squared_error', optimizer=opt)
              #metrics=['mae'])
history = model.fit(encoder_input_data, decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=epochs,
          validation_split=0.2)
# Save model
#model.save('audio_s2s.h5')

#find time required for training
stop_time = time.perf_counter()
print(f"Training done in {stop_time - start_time:0.4f} seconds")


# list all data in history
print(history.history.keys())

def evaluate(input_seq):
    result = ''
    enc_out, encoder_hidden_states = inference_encoder_model.predict(input_seq)
    print("encoder_hidden_states shape", encoder_hidden_states.shape)
    decoder_states_inference =  Input(shape=(units,)) 
    print("decoder_states_inference shape: ", decoder_states_inference)
    output, state = decoder_gru(decoder_inputs, initial_state = decoder_states_inference)
    output = decoder_dense(output)

    inference_decoder_model = Model([decoder_inputs, decoder_states_inference],
                                    [output, state])

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_of_features))
    #set the first frequency as the start input
    target_seq[0,0,:] = 0

    decoded_spec = np.zeros((1, 1, decoder_target_data.shape[2]))
    for t in range(max_length):
        #predictions, dec_hidden = decoder(target_seq.astype(np.float32), dec_hidden, enc_out)
        #print("predictions: ", predictions)
        #print("type",  type(predictions))
        #print("prediction shape: ",predictions.shape)
        #print("num of dimensions: ", (np.asarray(predictions)).ndim)
        predictions, dec_hidden = inference_decoder_model.predict([target_seq, encoder_hidden_states])

        print("prediction shape: ",predictions.shape)
        decoded_spec = np.append(decoded_spec, predictions, axis=1)
        index = decoded_spec.shape[1]
        if (decoded_spec[0,index-1,0] == 0):
            return decoded_spec

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_of_features))
        target_seq= predictions
        # Update states
        decoder_states_inference = dec_hidden
    print("decoded_spec shape: ", decoded_spec.shape)
    return decoded_spec
#DIR ="D:/UB_MS/Thesis/Seprating speech signals/DataSet/set/test/"
DIR = "D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/noisy_test/"
for i in range (1000):
    #s, sr = librosa.load(NOISY_SIGNALS_DIR +'clean' + str(i) + '_n1.wav',sr = sampling_frequency)

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

    librosa.output.write_wav('D:/UB_MS/Thesis/Seprating speech signals/DataSet/Results/one_noise_data/attention/256/enhanced_' + str(i) + '.wav', x_reconstruct, sr= sampling_frequency)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Attention-based encoder-decoder model loss ')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





