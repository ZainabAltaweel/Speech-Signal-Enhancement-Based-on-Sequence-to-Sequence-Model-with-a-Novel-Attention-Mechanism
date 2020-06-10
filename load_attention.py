import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math
import pickle
import librosa
from keras import backend as K
from keras.models import Model
from keras.layers import Input, GRU, Dense, Multiply, Softmax, Lambda, Concatenate
from keras_self_attention import SeqSelfAttention
from keras.models import load_model
import time
import sys
import tensorflow as tf
from BahdanauAttention import*
from tensorflow.python.client import device_lib

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
NOISY_SIGNALS_DIR = 'C:/Users/Zainab Altaweel/source/repos/RNN/RNN/data_set/noisy_audio_test/'

BATCH_SIZE = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
units = 256  # Latent dimensionality of the encoding space.

n_file = 10
max_length = 100 #max length in time domain
num_of_features = 513
hop_length = 512
n_fft = 1024
sampling_frequency = 16000


noisy_data_set = np.zeros((n_file, max_length, num_of_features), dtype='uint8')
clean_data_set = np.zeros((n_file, max_length, num_of_features), dtype='uint8')

hf = h5py.File(NOISY_SIGNALS_DIR + 'data10.h5', 'r')
print(hf.keys())
noisy_data_set = hf.get('noisy_signals')
noisy_data_set = (np.array(noisy_data_set)).astype(np.float32)
clean_data_set = hf.get('clean_signals')
clean_data_set = (np.array(clean_data_set)).astype(np.float32)

encoder_input_data = noisy_data_set
decoder_target_data = clean_data_set
decoder_input_data = np.zeros((n_file, 1, num_of_features))
decoder_input_data[:,0,:] = 0

encoder_input_data = noisy_data_set
decoder_target_data = clean_data_set


print("noisy data shape: ", noisy_data_set.shape)
print("clean data shape: ", clean_data_set.shape)
BUFFER_SIZE = noisy_data_set.shape[0]
print("buffer size: ", BUFFER_SIZE)

steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
print("steps_per_epoch: ", steps_per_epoch)
dataset = tf.data.Dataset.from_tensor_slices((noisy_data_set, clean_data_set)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


#---------------------------------My Model----------------------------------#
#---------------------------------Encoder-----------------------------------#
encoder_inputs = Input(shape=( max_length, num_of_features))
print("input shape: ", encoder_inputs.shape)
encoder_gru = GRU(units, return_sequences=True, return_state=True)
enc_output, enc_hidden = encoder_gru(encoder_inputs)

print ('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))

#----------------------------------Attention-------------------------------#
W1 = Dense(units)
W2 = Dense(units)
V = Dense(1)
SM = Softmax(axis=1)
context_vector_mul = Multiply()
reduce_sum = Lambda(lambda x: tf.reduce_sum(x, axis=1))
concat = Concatenate(axis=-1)

query_with_time_axis = tf.expand_dims(enc_hidden, 1)
score = V(tf.nn.tanh(
        W1(query_with_time_axis) + W2(enc_output)))
#attention_weights = tf.nn.softmax(score, axis=1)
attention_weights = SM(score)

# context_vector shape after sum == (batch_size, hidden_size)
context_vector = context_vector_mul([attention_weights , enc_output])

#context_vector = tf.reduce_sum(context_vector, axis=1)
context_vector = reduce_sum(context_vector)

#----------------------------------Decoder-----------------------------------#

decoder_inputs = Input(shape=(1, num_of_features))
decoder_gru = GRU(units, return_sequences=True, return_state=True)
decoder_dense = Dense(num_of_features)
multiply_layer = Multiply()
all_outputs = []
inputs = decoder_inputs
states = enc_hidden
for _ in range(max_length):
    # Run the decoder on one timestep
    decoder_outputs = concat([tf.expand_dims(context_vector, 1), inputs])
    decoder_outputs, decoder_state = decoder_gru(inputs, initial_state= states)
    decoder_outputs = decoder_dense(decoder_outputs)
    # Store the current prediction (we will concatenate all predictions later)
    all_outputs.append(decoder_outputs)
    # Reinject the outputs as inputs for the next loop iteration
    # as well as update the states
    inputs = decoder_outputs
    states = decoder_state

# Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)
decoder_outputs = multiply_layer([encoder_inputs, decoder_outputs])

print ('Decoder output shape: (batch_size, vocab size) {}'.format(decoder_outputs.shape))

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

start_time = time.perf_counter()
model.summary()
model.compile(optimizer='Adam', loss='mean_squared_error',
              metrics=['mae'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('audio_s2s.h5')

#find time required for training
stop_time = time.perf_counter()
print(f"Training done in {stop_time - start_time:0.4f} seconds")


#----------------------------------Inference Model---------------------------#

inference_encoder_model = Model(encoder_inputs, [enc_output, enc_hidden])
def evaluate_teacher_forcing(input_seq):
    #initial_hidden = encoder.initialize_hidden_state()
    result = ''
    enc_out, encoder_hidden_states = inference_encoder_model.predict(input_seq)
    print("encoder_hidden_states shape", encoder_hidden_states.shape)
    decoder_states_inference =  Input(shape=(units,)) 
    print("decoder_states_inference shape: ", decoder_states_inference)
    output, state = decoder_gru(decoder_inputs, initial_state = decoder_states_inference)
    output = decoder_dense(output)
    output = multiply_layer([encoder_inputs, output])
    inference_decoder_model = Model([encoder_inputs, decoder_inputs, decoder_states_inference],
                                    [output, state])

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_of_features))
    #set the first frequency as the start input
    target_seq[0,0,:] = 0

    decoded_spec = np.zeros((1, 1, decoder_target_data.shape[2]))
    for t in range(max_length):

        predictions, dec_hidden = inference_decoder_model.predict([input_seq, target_seq, encoder_hidden_states])
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

def evaluate(input_seq):
    decoded_spec = model.predict([input_seq, decoder_input_data])
    return decoded_spec

for i in range (30):
    stft_seq=np.zeros((1, decoder_target_data.shape[1], decoder_target_data.shape[2]))
    s, sr = librosa.load(NOISY_SIGNALS_DIR +'clean' + str(i) + '_n1.wav',sr = sampling_frequency)
    print("sr= ", sr)
    stft = librosa.stft(s, n_fft=n_fft, hop_length=hop_length)
    stft_len = stft.shape[1]
    stft_abs = np.abs(stft)
    stft_abs = np.pad(stft_abs, ((0,0),(0, max_length - stft_len)), 'constant')
    stft_seq[0]=stft_abs.T

    print("stft seq shape: ", stft_seq.shape)
    print(stft_seq[0])
    decoded_audio = evaluate(stft_seq)
    decoded_output = np.asarray(decoded_audio)
    print(decoded_audio[0])

    print("decoded_output",decoded_output[0].shape)
    s_pred = librosa.istft(decoded_output[0].T, win_length = 1024, hop_length = 512)
    signal_len = s_pred.shape[0]

    x_reconstruct = reconstruct_signal_griffin_lim(decoded_output[0], n_fft, hop_length,signal_len, 300)

    librosa.output.write_wav('C:/Users/Zainab Altaweel/source/repos/RNN/RNN/DataSet/GRU_Attention0323/gru_seq2seq_attention_0323_' + str(i) + '.wav', x_reconstruct, sr= sampling_frequency)
