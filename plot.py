import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

hop_length = 512
n_fft = 1024

x1 = [128, 256, 512]
y1 = [ 1.631, 1.658, 1.676]
y2 = [82.42, 83.06, 83.47]
# plotting the line 1 points 

plt.subplot(1, 2, 1) 
plt.plot(x1, y1, label = "PESQ ") 
plt.xlabel('Number of neurons') 
plt.ylabel('PESQ') 


plt.subplot(1, 2, 2) 
plt.plot(x1, y2, label = "STOI")
plt.xlabel('Number of neurons') 
plt.ylabel('STOI') 
 
  

y, sr = librosa.load("D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/noisy_test/noisy462.wav", sr= 16000)
plt.figure(figsize=(12, 8))
D1 = np.abs(librosa.stft(y, n_fft=n_fft,  
                        hop_length=hop_length))
D1 = librosa.amplitude_to_db(D1, ref=np.max)

y, sr = librosa.load("D:/UB_MS/Thesis/Seprating speech signals/DataSet/set/test_clean/clean462.wav", sr= 16000)
plt.figure(figsize=(12, 8))
D2 = np.abs(librosa.stft(y, n_fft=n_fft,  
                        hop_length=hop_length))
D2 = librosa.amplitude_to_db(D2, ref=np.max)

y, sr = librosa.load("D:/UB_MS/Thesis/Seprating speech signals/DataSet/Results/one_noise_data/seq2seq/256/enhanced_462.wav", sr= 16000)
plt.figure(figsize=(12, 8))
D3 = np.abs(librosa.stft(y, n_fft=n_fft,  
                        hop_length=hop_length))
D3= librosa.amplitude_to_db(D3, ref=np.max)

y, sr = librosa.load("D:/UB_MS/Thesis/Seprating speech signals/DataSet/Results/one_noise_data/attention/256/enhanced_462.wav", sr= 16000)
plt.figure(figsize=(12, 8))
D4 = np.abs(librosa.stft(y, n_fft=n_fft,  
                        hop_length=hop_length))
D4= librosa.amplitude_to_db(D4, ref=np.max) 


plt.subplot(2, 2, 1)
librosa.display.specshow(D2, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Original clean signal')

plt.subplot(2, 2, 2)
librosa.display.specshow(D1, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Original noisy signal')

plt.subplot(2, 2, 3)
librosa.display.specshow(D3, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Enhanced signal using enc-dec-GRU')

plt.subplot(2, 2, 4)
librosa.display.specshow(D4, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Enhanced signal using att-enc-dec-GRU')
plt.show()
